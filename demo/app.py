# Copyright (c) 2024 Ye Liu. Licensed under the BSD-3-Clause license.

import html
import json
import os
import random
import time
from functools import partial
from threading import Thread

import gradio as gr
import nncore
import torch
from huggingface_hub import snapshot_download
from transformers import TextIteratorStreamer

from videomind.constants import GROUNDER_PROMPT, PLANNER_PROMPT, VERIFIER_PROMPT
from videomind.dataset.utils import process_vision_info
from videomind.model.builder import build_model
from videomind.utils.io import get_duration
from videomind.utils.parser import parse_query, parse_span

BASE_MODEL = 'model_zoo/Qwen2-VL-2B-Instruct'
BASE_MODEL_HF = 'Qwen/Qwen2-VL-2B-Instruct'

MODEL = 'model_zoo/VideoMind-2B'
MODEL_HF = 'yeliudev/VideoMind-2B'

TITLE = 'VideoMind: A Chain-of-LoRA Agent for Long Video Reasoning'

TITLE_MD = f'<h1 align="center">💡 {TITLE}</h1>'
DESCRIPTION_MD = """VideoMind is a multi-modal agent framework that enhances video reasoning by emulating *human-like* processes, such as *breaking down tasks*, *localizing and verifying moments*, and *synthesizing answers*. This approach addresses the unique challenges of temporal-grounded reasoning in a progressive strategy. Please find more details at our <a href="https://videomind.github.io/" target="_blank">Project Page</a>, <a href="https://arxiv.org/abs/2503.13444" target="_blank">Tech Report</a> and <a href="https://github.com/yeliudev/VideoMind" target="_blank">GitHub Repo</a>."""  # noqa

# yapf:disable
EXAMPLES = [
    ('data/4167294363.mp4', 'Why did the old man stand up?', ['pla', 'gnd', 'ver', 'ans']),
    ('data/5012237466.mp4', 'How does the child in stripes react about the fountain?', ['pla', 'gnd', 'ver', 'ans']),
    ('data/13887487955.mp4', 'What did the excavator do after it pushed the cement forward?', ['pla', 'gnd', 'ver', 'ans']),
    ('data/5188348585.mp4', 'What did the person do before pouring the liquor?', ['pla', 'gnd', 'ver', 'ans']),
    ('data/4766274786.mp4', 'What did the girl do after the baby lost the balloon?', ['pla', 'gnd', 'ver', 'ans']),
    ('data/4742652230.mp4', 'Why is the girl pushing the boy only around the toy but not to other places?', ['pla', 'gnd', 'ver', 'ans']),
    ('data/9383140374.mp4', 'How does the girl in pink control the movement of the claw?', ['pla', 'gnd', 'ver', 'ans']),
    ('data/10309844035.mp4', 'Why are they holding up the phones?', ['pla', 'gnd', 'ver', 'ans']),
    ('data/pA6Z-qYhSNg_60.0_210.0.mp4', 'Different types of meat products are being cut, shaped and prepared', ['gnd', 'ver']),
    ('data/UFWQKrcbhjI_360.0_510.0.mp4', 'A man talks to the camera whilst walking along a roadside in a rural area', ['gnd', 'ver']),
    ('data/RoripwjYFp8_210.0_360.0.mp4', 'A woman wearing glasses eating something at a street market', ['gnd', 'ver']),
    ('data/h6QKDqomIPk_210.0_360.0.mp4', 'A toddler sits in his car seat, holding his yellow tablet', ['gnd', 'ver']),
    ('data/Z3-IZ3HAmIA_60.0_210.0.mp4', 'A view from the window as the plane accelerates and takes off from the runway', ['gnd', 'ver']),
    ('data/yId2wIocTys_210.0_360.0.mp4', "Temporally locate the visual content mentioned in the text query 'kids exercise in front of parked cars' within the video.", ['pla', 'gnd', 'ver']),
    ('data/rrTIeJRVGjg_60.0_210.0.mp4', "Localize the moment that provides relevant context about 'man stands in front of a white building monologuing'.", ['pla', 'gnd', 'ver']),
    ('data/DTInxNfWXVc_210.0_360.0.mp4', "Find the video segment that corresponds to the given textual query 'man with headphones talking'.", ['pla', 'gnd', 'ver']),
]
# yapf:enable

CSS = """button .box { text-align: left }"""

JS = """
function init() {
    var info = document.getElementById('role').querySelectorAll('[class^="svelte"]')[1]
    info.innerHTML = info.innerHTML.replace(/&lt;/g, '<').replace(/&gt;/g, '>')
}
"""


class CustomStreamer(TextIteratorStreamer):

    def put(self, value):
        if len(value.shape) > 1 and value.shape[0] > 1:
            raise ValueError('TextStreamer only supports batch size 1')
        elif len(value.shape) > 1:
            value = value[0]

        if self.skip_prompt and self.next_tokens_are_prompt:
            self.next_tokens_are_prompt = False
            return

        self.token_cache.extend(value.tolist())

        # force skipping eos token
        if self.token_cache[-1] == self.tokenizer.eos_token_id:
            self.token_cache = self.token_cache[:-1]

        text = self.tokenizer.decode(self.token_cache, **self.decode_kwargs)

        # cache decoded text for future use
        self.text_cache = text

        if text.endswith('\n'):
            printable_text = text[self.print_len:]
            self.token_cache = []
            self.print_len = 0
        elif len(text) > 0 and self._is_chinese_char(ord(text[-1])):
            printable_text = text[self.print_len:]
            self.print_len += len(printable_text)
        else:
            printable_text = text[self.print_len:text.rfind(' ') + 1]
            self.print_len += len(printable_text)

        self.on_finalized_text(printable_text)


def seconds_to_hms(seconds):
    hours, remainder = divmod(round(seconds), 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02}:{minutes:02}:{seconds:02}'


def enable_btns():
    return (gr.Button(interactive=True), ) * 3


def disable_btns():
    return (gr.Button(interactive=False), ) * 3


def update_placeholder(role):
    placeholder = 'Ask a question about the video...' if 'ans' in role else 'Write a query to search for a moment...'
    return gr.Textbox(placeholder=placeholder)


def main(video, prompt, role, temperature, max_new_tokens, model, processor, streamer, device):
    history = []

    if not video:
        gr.Warning('Please upload a video or click [Random] to sample one.')
        return history

    if not prompt:
        gr.Warning('Please provide a prompt or click [Random] to sample one.')
        return history

    if 'gnd' not in role and 'ans' not in role:
        gr.Warning('Please at least select Grounder or Answerer.')
        return history

    if 'ver' in role and 'gnd' not in role:
        gr.Warning('Verifier cannot be used without Grounder.')
        return history

    if 'pla' in role and 'gnd' not in role and 'ver' not in role:
        gr.Warning('Planner can only be used with Grounder and Verifier.')
        return history

    history.append({'role': 'user', 'content': prompt})
    yield history

    duration = get_duration(video)

    # do grounding and answering by default
    do_grounding = True
    do_answering = True

    # initialize grounding query as prompt
    query = prompt

    if 'pla' in role:
        text = PLANNER_PROMPT.format(prompt)

        history.append({
            'metadata': {
                'title': '🗺️ Working as Planner...'
            },
            'role': 'assistant',
            'content': f'##### Planner Prompt:\n\n{html.escape(text)}\n\n##### Planner Response:\n\n...'
        })
        yield history

        start_time = time.perf_counter()

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video,
                'num_threads': 1,
                'min_pixels': 36 * 28 * 28,
                'max_pixels': 64 * 28 * 28,
                'max_frames': 100,
                'fps': 1.0
            }, {
                'type': 'text',
                'text': text
            }]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)

        images, videos = process_vision_info(messages)
        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(device)

        model.base_model.disable_adapter_layers()
        model.base_model.enable_adapter_layers()
        model.set_adapter('planner')

        generation_kwargs = dict(
            **data,
            streamer=streamer,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=max_new_tokens)

        t = Thread(target=model.generate, kwargs=generation_kwargs)
        t.start()

        skipped = False
        for i, text in enumerate(streamer):
            if text and not skipped:
                history[-1]['content'] = history[-1]['content'].rstrip('.')
                skipped = True
            history[-1]['content'] += text
            yield history

        elapsed_time = round(time.perf_counter() - start_time, 1)
        history[-1]['metadata']['title'] += f' ({elapsed_time} seconds)'
        yield history

        try:
            parsed = json.loads(streamer.text_cache)
            action = parsed[0] if isinstance(parsed, list) else parsed
            if action['type'].lower() == 'grounder' and action['value']:
                query = action['value']
            elif action['type'].lower() == 'answerer':
                do_grounding = False
                do_answering = True
        except Exception:
            pass

        response = 'After browsing the video and the question. My plan to figure out the answer is as follows:\n'
        step_idx = 1
        if 'gnd' in role and do_grounding:
            response += f'\n{step_idx}. Localize the relevant moment in this video using the query "<span style="color:red">{query}</span>".'
            step_idx += 1
        if 'ver' in role and do_grounding:
            response += f'\n{step_idx}. Verify the grounded moments one-by-one and select the best cancdidate.'
            step_idx += 1
        if 'ans' in role and do_answering:
            if step_idx > 1:
                response += f'\n{step_idx}. Crop the video segment and zoom-in to higher resolution.'
            else:
                response += f'\n{step_idx}. Analyze the whole video directly without cropping.'

        history.append({'role': 'assistant', 'content': ''})
        for i, text in enumerate(response.split(' ')):
            history[-1]['content'] += ' ' + text if i > 0 else text
            yield history

    if 'gnd' in role and do_grounding:
        query = parse_query(query)

        text = GROUNDER_PROMPT.format(query)

        history.append({
            'metadata': {
                'title': '🔍 Working as Grounder...'
            },
            'role': 'assistant',
            'content': f'##### Grounder Prompt:\n\n{html.escape(text)}\n\n##### Grounder Response:\n\n...'
        })
        yield history

        start_time = time.perf_counter()

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video,
                'num_threads': 1,
                'min_pixels': 36 * 28 * 28,
                'max_pixels': 64 * 28 * 28,
                'max_frames': 150,
                'fps': 1.0
            }, {
                'type': 'text',
                'text': text
            }]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(device)

        model.base_model.disable_adapter_layers()
        model.base_model.enable_adapter_layers()
        model.set_adapter('grounder')

        generation_kwargs = dict(
            **data,
            streamer=streamer,
            do_sample=temperature > 0,
            temperature=temperature if temperature > 0 else None,
            top_p=None,
            top_k=None,
            repetition_penalty=None,
            max_new_tokens=max_new_tokens)

        t = Thread(target=model.generate, kwargs=generation_kwargs)
        t.start()

        skipped = False
        for i, text in enumerate(streamer):
            if text and not skipped:
                history[-1]['content'] = history[-1]['content'].rstrip('.')
                skipped = True
            history[-1]['content'] += text
            yield history

        elapsed_time = round(time.perf_counter() - start_time, 1)
        history[-1]['metadata']['title'] += f' ({elapsed_time} seconds)'
        yield history

        if len(model.reg) > 0:
            # 1. extract timestamps and confidences
            blob = model.reg[0].cpu().float()
            pred, conf = blob[:, :2] * duration, blob[:, -1].tolist()

            # 2. clamp timestamps
            pred = pred.clamp(min=0, max=duration)

            # 3. sort timestamps
            inds = (pred[:, 1] - pred[:, 0] < 0).nonzero()[:, 0]
            pred[inds] = pred[inds].roll(1)

            # 4. convert timestamps to list
            pred = pred.tolist()
        else:
            if 'ver' in role:
                pred = [[i * duration / 6, (i + 2) * duration / 6] for i in range(5)]
                conf = [0] * 5
            else:
                pred = [[0, duration]]
                conf = [0]

        response = 'The candidate moments and confidence scores are as follows:\n'
        response += '\n| ID | Start Time | End Time | Confidence |'
        response += '\n| :-: | :-: | :-: | :-: |'

        # using top-5 predictions
        for i, (p, c) in enumerate(zip(pred[:5], conf[:5])):
            response += f'\n| {i} | {seconds_to_hms(p[0])} | {seconds_to_hms(p[1])} | {c:.2f} |'

        response += f'\n\nTherefore, the target moment might happens from <span style="color:red">{seconds_to_hms(pred[0][0])}</span> to <span style="color:red">{seconds_to_hms(pred[0][1])}</span>.'

        history.append({'role': 'assistant', 'content': ''})
        for i, text in enumerate(response.split(' ')):
            history[-1]['content'] += ' ' + text if i > 0 else text
            yield history

    if 'ver' in role and do_grounding:
        text = VERIFIER_PROMPT.format(query)

        history.append({
            'metadata': {
                'title': '📊 Working as Verifier...'
            },
            'role': 'assistant',
            'content': f'##### Verifier Prompt:\n\n{html.escape(text)}\n\n##### Verifier Response:\n\n...'
        })
        yield history

        start_time = time.perf_counter()

        # using top-5 predictions
        prob = []
        for i, cand in enumerate(pred[:5]):
            s0, e0 = parse_span(cand, duration, 2)
            offset = (e0 - s0) / 2
            s1, e1 = parse_span([s0 - offset, e0 + offset], duration)

            # percentage of s0, e0 within s1, e1
            s = (s0 - s1) / (e1 - s1)
            e = (e0 - s1) / (e1 - s1)

            messages = [{
                'role':
                'user',
                'content': [{
                    'type': 'video',
                    'video': video,
                    'num_threads': 1,
                    'video_start': s1,
                    'video_end': e1,
                    'min_pixels': 36 * 28 * 28,
                    'max_pixels': 64 * 28 * 28,
                    'max_frames': 64,
                    'fps': 2.0
                }, {
                    'type': 'text',
                    'text': text
                }]
            }]

            text = processor.apply_chat_template(messages, add_generation_prompt=True)
            images, videos = process_vision_info(messages)
            data = processor(text=[text], images=images, videos=videos, return_tensors='pt')

            # ===== insert segment start/end tokens =====
            video_grid_thw = data['video_grid_thw'][0]
            num_frames, window = int(video_grid_thw[0]), int(video_grid_thw[1] * video_grid_thw[2] / 4)
            assert num_frames * window * 4 == data['pixel_values_videos'].size(0)

            pos_s, pos_e = round(s * num_frames), round(e * num_frames)
            pos_s, pos_e = min(max(0, pos_s), num_frames), min(max(0, pos_e), num_frames)
            assert pos_s <= pos_e, (num_frames, s, e)

            base_idx = torch.nonzero(data['input_ids'][0] == model.config.vision_start_token_id).item()
            pos_s, pos_e = pos_s * window + base_idx + 1, pos_e * window + base_idx + 2

            input_ids = data['input_ids'][0].tolist()
            input_ids.insert(pos_s, model.config.seg_s_token_id)
            input_ids.insert(pos_e, model.config.seg_e_token_id)
            data['input_ids'] = torch.LongTensor([input_ids])
            data['attention_mask'] = torch.ones_like(data['input_ids'])
            # ===========================================

            data = data.to(device)

            model.base_model.disable_adapter_layers()
            model.base_model.enable_adapter_layers()
            model.set_adapter('verifier')

            with torch.inference_mode():
                logits = model(**data).logits[0, -1].softmax(dim=-1)

            # NOTE: magic numbers here
            # In Qwen2-VL vocab: 9454 -> Yes, 2753 -> No
            score = (logits[9454] - logits[2753]).sigmoid().item()
            prob.append(score)

            if i == 0:
                history[-1]['content'] = history[-1]['content'].rstrip('.')[:-1]

            response = f'\nCandidate ID {i}: P(Yes) = {score:.2f}'
            for j, text in enumerate(response.split(' ')):
                history[-1]['content'] += ' ' + text if j > 0 else text
                yield history

        elapsed_time = round(time.perf_counter() - start_time, 1)
        history[-1]['metadata']['title'] += f' ({elapsed_time} seconds)'
        yield history

        ranks = torch.Tensor(prob).argsort(descending=True).tolist()

        prob = [prob[idx] for idx in ranks]
        pred = [pred[idx] for idx in ranks]
        conf = [conf[idx] for idx in ranks]

        response = 'After verification, the candidate moments are re-ranked as follows:\n'
        response += '\n| ID | Start Time | End Time | Score |'
        response += '\n| :-: | :-: | :-: | :-: |'

        ids = list(range(len(ranks)))
        for r, p, c in zip(ranks, pred, prob):
            response += f'\n| {ids[r]} | {seconds_to_hms(p[0])} | {seconds_to_hms(p[1])} | {c:.2f} |'

        response += f'\n\nTherefore, the target moment should be from <span style="color:red">{seconds_to_hms(pred[0][0])}</span> to <span style="color:red">{seconds_to_hms(pred[0][1])}</span>.'

        history.append({'role': 'assistant', 'content': ''})
        for i, text in enumerate(response.split(' ')):
            history[-1]['content'] += ' ' + text if i > 0 else text
            yield history

    if 'ans' in role and do_answering:
        text = f'{prompt} Please think step by step and provide your response.'

        history.append({
            'metadata': {
                'title': '📝 Working as Answerer...'
            },
            'role': 'assistant',
            'content': f'##### Answerer Prompt:\n\n{html.escape(text)}\n\n##### Answerer Response:\n\n...'
        })
        yield history

        start_time = time.perf_counter()

        # choose the potential best moment
        selected = pred[0] if 'gnd' in role and do_grounding else [0, duration]
        s, e = parse_span(selected, duration, 32)

        messages = [{
            'role':
            'user',
            'content': [{
                'type': 'video',
                'video': video,
                'num_threads': 1,
                'video_start': s,
                'video_end': e,
                'min_pixels': 128 * 28 * 28,
                'max_pixels': 256 * 28 * 28,
                'max_frames': 32,
                'fps': 2.0
            }, {
                'type': 'text',
                'text': text
            }]
        }]

        text = processor.apply_chat_template(messages, add_generation_prompt=True)
        images, videos = process_vision_info(messages)
        data = processor(text=[text], images=images, videos=videos, return_tensors='pt')
        data = data.to(device)

        with model.disable_adapter():
            generation_kwargs = dict(
                **data,
                streamer=streamer,
                do_sample=temperature > 0,
                temperature=temperature if temperature > 0 else None,
                top_p=None,
                top_k=None,
                repetition_penalty=None,
                max_new_tokens=max_new_tokens)

            t = Thread(target=model.generate, kwargs=generation_kwargs)
            t.start()

            skipped = False
            for i, text in enumerate(streamer):
                if text and not skipped:
                    history[-1]['content'] = history[-1]['content'].rstrip('.')
                    skipped = True
                history[-1]['content'] += text
                yield history

        elapsed_time = round(time.perf_counter() - start_time, 1)
        history[-1]['metadata']['title'] += f' ({elapsed_time} seconds)'
        yield history

        if 'gnd' in role and do_grounding:
            response = f'After zooming in and analyzing the target moment, I finalize my answer: <span style="color:green">{streamer.text_cache}</span>'
        else:
            response = f'After watching the whole video, my answer is: <span style="color:green">{streamer.text_cache}</span>'

        history.append({'role': 'assistant', 'content': ''})
        for i, text in enumerate(response.split(' ')):
            history[-1]['content'] += ' ' + text if i > 0 else text
            yield history


if __name__ == '__main__':
    if not nncore.is_dir(BASE_MODEL):
        snapshot_download(BASE_MODEL_HF, local_dir=BASE_MODEL)

    if not nncore.is_dir(MODEL):
        snapshot_download(MODEL_HF, local_dir=MODEL)

    print('Initializing role *grounder*')
    model, processor = build_model(MODEL)

    print('Initializing role *planner*')
    model.load_adapter(nncore.join(MODEL, 'planner'), adapter_name='planner')

    print('Initializing role *verifier*')
    model.load_adapter(nncore.join(MODEL, 'verifier'), adapter_name='verifier')

    streamer = CustomStreamer(processor.tokenizer, skip_prompt=True)

    device = next(model.parameters()).device

    main = partial(main, model=model, processor=processor, streamer=streamer, device=device)

    path = os.path.dirname(os.path.realpath(__file__))

    chat = gr.Chatbot(
        type='messages',
        height='70vh',
        avatar_images=[f'{path}/assets/user.png', f'{path}/assets/bot.png'],
        placeholder='A conversation with VideoMind',
        label='VideoMind')

    prompt = gr.Textbox(label='Text Prompt', placeholder='Ask a question about the video...')

    with gr.Blocks(title=TITLE, css=CSS, js=JS) as demo:
        gr.Markdown(TITLE_MD)
        gr.Markdown(DESCRIPTION_MD)

        with gr.Row():
            with gr.Column(scale=3):
                video = gr.Video()

                with gr.Group():
                    role = gr.CheckboxGroup(
                        choices=[('🗺️ Planner', 'pla'), ('🔍 Grounder', 'gnd'), ('📊 Verifier', 'ver'),
                                 ('📝 Answerer', 'ans')],
                        value=['pla', 'gnd', 'ver', 'ans'],
                        interactive=True,
                        elem_id='role',
                        label='Roles',
                        info='Select the role(s) you would like to activate.')
                    role.change(update_placeholder, role, prompt)

                    with gr.Accordion(label='Hyperparameters', open=False):
                        temperature = gr.Slider(
                            0,
                            1,
                            value=0,
                            step=0.1,
                            interactive=True,
                            label='Temperature',
                            info='Higher value leads to more creativity and randomness (Default: 0)')
                        max_new_tokens = gr.Slider(
                            1,
                            1024,
                            value=256,
                            interactive=True,
                            label='Max Output Tokens',
                            info='The maximum number of output tokens for each role (Default: 256)')

                prompt.render()

                with gr.Row():
                    random_btn = gr.Button(value='🔮 Random')
                    random_btn.click(lambda: random.choice(EXAMPLES), None, [video, prompt, role])

                    reset_btn = gr.ClearButton([video, prompt, chat], value='🗑️ Reset')
                    reset_btn.click(lambda: (['pla', 'gnd', 'ver', 'ans'], 0, 256), None,
                                    [role, temperature, max_new_tokens])

                    submit_btn = gr.Button(value='🚀 Submit', variant='primary')
                    submit_ctx = submit_btn.click(disable_btns, None, [random_btn, reset_btn, submit_btn])
                    submit_ctx = submit_ctx.then(main, [video, prompt, role, temperature, max_new_tokens], chat)
                    submit_ctx.then(enable_btns, None, [random_btn, reset_btn, submit_btn])

            with gr.Column(scale=5):
                chat.render()

        demo.launch(server_name='0.0.0.0')
