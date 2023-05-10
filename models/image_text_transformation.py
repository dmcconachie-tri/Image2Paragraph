from models.blip2_model import ImageCaptioning
from models.grit_model import DenseCaptioning
from models.gpt_model import ImageToText
from models.controlnet_model import TextToImage
from models.region_semantic import RegionSemantic
from utils.util import read_image_width_height, display_images_and_text, resize_long_edge
import argparse
from PIL import Image
import base64
from io import BytesIO
import os
import pprint
import time

def pil_image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str


class ImageTextTransformation:
    def __init__(self, args):
        # Load your big model here
        self.args = args
        self.init_models()
        self.ref_image = None

    def init_models(self):
        start = time.time()
        openai_key = os.environ['OPENAI_KEY']
        pprint.pprint(vars(self.args), sort_dicts=False)
        printwidth = 80

        print('\033[1;34m' + "Welcome to the Image2Paragraph toolbox...".center(printwidth, '-') + '\033[0m')
        print('\033[1;33m' + "Initializing models...".center(printwidth, '-') + '\033[0m')
        print('\033[1;31m' + "This is time-consuming, please wait...".center(printwidth, '-') + '\033[0m')

        print('\033[1;33m' + f"{(time.time() - start):.2f} Initializing ImageCaptioning {self.args.captioner_base_model}...".center(printwidth, '-') + '\033[0m')
        self.image_caption_model = ImageCaptioning(
            device=self.args.image_caption_device,
            captioner_base_model=self.args.captioner_base_model,
        )
        print('\033[1;33m' + f"{(time.time() - start):.2f} Initializing DenseCaptioning...".center(printwidth, '-') + '\033[0m')
        self.dense_caption_model = DenseCaptioning(
            device=self.args.dense_caption_device,
        )
        print('\033[1;33m' + f"{(time.time() - start):.2f} Initializing ImageToText {self.args.gpt_version}...".center(printwidth, '-') + '\033[0m')
        self.gpt_model = ImageToText(
            openai_key,
            gpt_version=self.args.gpt_version,
        )
        print('\033[1;33m' + f"{(time.time() - start):.2f} Initializing TextToImage (controlnet)...".center(printwidth, '-') + '\033[0m')
        self.controlnet_model = TextToImage(
            device=self.args.contolnet_device,
        )

        print('\033[1;33m' + f"{(time.time() - start):.2f} Initializing RegionSemantic {self.args.region_classify_model}...".center(printwidth, '-') + '\033[0m')
        self.region_semantic_model = RegionSemantic(
            device=self.args.semantic_segment_device,
            image_caption_model=self.image_caption_model,
            region_classify_model=self.args.region_classify_model,
            sam_arch=self.args.sam_arch,
        )
        print('\033[1;32m' + f"{(time.time() - start):.2f} Model initialization finished!".center(printwidth, '-') + '\033[0m')

    def image_to_text(self, img_src):
        # the information to generate paragraph based on the context
        self.ref_image = Image.open(img_src)
        # resize image to long edge 384
        self.ref_image = resize_long_edge(self.ref_image, 384)
        width, height = read_image_width_height(img_src)
        print(self.args)
        if self.args.image_caption:
            image_caption = self.image_caption_model.image_caption(img_src)
        else:
            image_caption = " "
        if self.args.dense_caption:
            dense_caption = self.dense_caption_model.image_dense_caption(img_src)
        else:
            dense_caption = " "
        if self.args.semantic_segment:
            region_semantic = self.region_semantic_model.region_semantic(img_src)
        else:
            region_semantic = " "
        generated_text = self.gpt_model.paragraph_summary_with_gpt(
            image_caption, dense_caption, region_semantic, width, height)
        return generated_text

    def text_to_image(self, text):
        generated_image = self.controlnet_model.text_to_image(text, self.ref_image)
        return generated_image

    def text_to_image_retrieval(self, text):
        pass

    def image_to_text_retrieval(self, image):
        pass
