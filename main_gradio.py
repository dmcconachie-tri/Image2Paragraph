import gradio as gr
import cv2
import numpy as np
from PIL import Image
import base64
from io import BytesIO
from models.image_text_transformation import ImageTextTransformation
import argparse
import warnings

parser = argparse.ArgumentParser()
# parser.add_argument('--image_src', default='examples/1.jpg')
# parser.add_argument('--out_image_name', default='output/1_result.jpg')
parser.add_argument(
    '--gpt_version',
    choices=['gpt-3.5-turbo', 'gpt4'],
    default='gpt-3.5-turbo',
)
parser.add_argument(
    '--image_caption',
    action='store_true',
    dest='image_caption',
    default=True,
    help='Set this flag to True if you want to use BLIP2 Image Caption',
)
parser.add_argument(
    '--dense_caption',
    action='store_true',
    dest='dense_caption',
    default=True,
    help='Set this flag to True if you want to use Dense Caption',
)
parser.add_argument(
    '--semantic_segment',
    action='store_true',
    dest='semantic_segment',
    default=True,
    help='Set this flag to True if you want to use semantic segmentation',
)

parser.add_argument(
    '--sam_arch',
    choices=['vit_b', 'vit_l', 'vit_h'],
    dest='sam_arch',
    default='vit_h',
    help='vit_b is the default model (fast but not accurate), vit_l and vit_h are larger models',
)
parser.add_argument(
    '--captioner_base_model',
    choices=['blip', 'blip2'],
    dest='captioner_base_model',
    default='blip2',
    help='blip2 requires 15G GPU memory, blip requires 6G GPU memory',
)
parser.add_argument(
    '--region_classify_model',
    choices=['ssa', 'edit_anything'],
    dest='region_classify_model',
    default='ssa',
    help='Select the region classification model: edit anything is ten times faster than ssa, but less accurate.',
)

backends = ['cuda', 'cuda:0', 'cuda:1', 'cpu']
parser.add_argument(
    '--image_caption_device',
    choices=backends,
    default='cuda:0',
    help=f'Select the device: {backends}, gpu memory larger than 14G is recommended',
)
parser.add_argument(
    '--dense_caption_device', choices=backends,
    default='cuda:0',
    help=f'Select the device: {backends}, < 6G GPU is not recommended>',
)
parser.add_argument(
    '--semantic_segment_device',
    choices=backends,
    default='cuda:1',
    help=f'Select the device: {backends}, gpu memory larger than 14G is recommended. '
         ' Make sure this model and image_caption model on same device. (Only relevant for EditAnything?)',
)
parser.add_argument(
    '--contolnet_device',
    choices=backends,
    default='cuda:0',
    help=f'Select the device: {backends}, <6G GPU is not recommended>',
)

args = parser.parse_args()

if args.gpt_version == 'gpt4':
    warnings.warn("gpt4 is not currently supported. Reverting to gpt-3.5-turbo")
    args.gpt_version = "gpt-3.5-turbo"

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def add_logo():
    with open("examples/logo.png", "rb") as f:
        logo_base64 = base64.b64encode(f.read()).decode()
    return logo_base64

def process_image(image_src, options, devices, processor):
    print(f"options: {options}")
    print(f"devices: {devices}")

    processor.args.image_caption = "Image Caption" in options
    processor.args.dense_caption = "Dense Caption" in options
    processor.args.semantic_segment = "Semantic Segment" in options

    # processor.args.image_caption_device = "cuda:1" if "cuda_ic" in devices else "cpu"
    # processor.args.dense_caption_device = "cuda" if "cuda_dc" in devices else "cpu"
    # processor.args.semantic_segment_device = "cuda:1" if "cuda_ss" in devices else "cpu"
    # processor.args.contolnet_device = "cuda:0" if "cuda_cn" in devices else "cpu"

    gen_text = processor.image_to_text(image_src)
    gen_image = processor.text_to_image(gen_text)
    gen_image_str = pil_image_to_base64(gen_image)
    # Combine the outputs into a single HTML output
    custom_output = f'''
    <h2>Image->Text->Image:</h2>
    <div style="display: flex; flex-wrap: wrap;">
        <div style="flex: 1;">
            <h3>Image2Text</h3>
            <p>{gen_text}</p>
        </div>
        <div style="flex: 1;">
            <h3>Text2Image</h3>
            <img src="data:image/jpeg;base64,{gen_image_str}" width="100%" />
        </div>
    </div>
    '''

    return custom_output

with warnings.catch_warnings():
    # Squelch the StableDiffusionControlNet warning about the safety checker.
    warnings.filterwarnings(action='ignore', message="disabled the safety checker")
    processor = ImageTextTransformation(args)

# Create Gradio input and output components
image_input = gr.inputs.Image(type='filepath', label="Input Image")
options_checkboxes = gr.CheckboxGroup(
    label="Options",
    choices=["Image Caption", "Dense Caption", "Semantic Segment"],
    value=["Image Caption", "Dense Caption", "Semantic Segment"],
)
device_checkboxes = gr.CheckboxGroup(
    label="Device, ic: image caption, dc: dense caption, ss: semantic segment, cn: controlnet",
    choices=["cuda_ic", "cuda_dc", "cuda_ss", "cuda_cn"],
    value=["cuda_ic", "cuda_dc", "cuda_ss", "cuda_cn"],
)


logo_base64 = add_logo()
# Create the title with the logo
title_with_logo = (
    f'<img src="data:image/jpeg;base64,{logo_base64}" width="400" '
    'style="vertical-align: middle;"> Understanding Image with Text'
)

# Create Gradio interface
interface = gr.Interface(
    fn=lambda image, options, devices: process_image(image, options, devices, processor),
    inputs=[image_input, options_checkboxes, device_checkboxes],
    outputs=gr.outputs.HTML(),
    title=title_with_logo,
    description="""
    This code support image to text transformation. Then the generated text can
    do retrieval, question answering et al to conduct zero-shot.
    \n Semantic segment is very slow in cpu, best use on gpu.
    \n If you have only 8GB Memory GPU, please set device as cpu for IC and SS.
    """
)

# Launch the interface
interface.launch()
