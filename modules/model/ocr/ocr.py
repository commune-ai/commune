import easyocr
import numpy as np
import cv2
from PIL import Image
import commune as c
import gradio as gr
from io import BytesIO
import base64
import os

languages = [
    ("English",	"en"),
    ("Abaza",	"abq"),
    ("Adyghe",	"ady"),
    ("Afrikaans",	"af"),
    ("Angika",	"ang"),
    ("Arabic",	"ar"),
    ("Assamese"",	as" ),
    ("Avar",	"ava"),
    ("Azerbaijani",	"az"),
    ("Belarusian",	"be"),
    ("Bulgarian",	"bg"),
    ("Bihari",	"bh"),
    ("Bhojpuri",	"bho"),
    ("Bengali",	"bn"),
    ("Bosnian",	"bs"),
    ("Simplified, Chinese"	"ch_sim"),
    ("Traditional, Chinese"	"ch_tra"),
    ("Chechen",	"che"),
    ("Czech",	"cs"),
    ("Welsh",	"cy"),
    ("Danish",	"da"),
    ("Dargwa",	"dar"),
    ("German",	"de"),
    ("Spanish",	"es"),
    ("Estonian",	"et"),
    ("Persian, (Farsi)"	"fa"),
    ("French",	"fr"),
    ("Irish",	"ga"),
    ("Goan, Konkani"	"gom"),
    ("Hindi",	"hi"),
    ("Croatian",	"hr"),
    ("Hungarian",	"hu"),
    ("Indonesian",	"id"),
    ("Ingush",	"inh"),
    ("Icelandic"",	is" ),
    ("Italian",	"it"),
    ("Japanese",	"ja"),
    ("Kabardian",	"kbd"),
    ("Kannada",	"kn"),
    ("Korean",	"ko"),
    ("Kurdish",	"ku"),
    ("Latin",	"la"),
    ("Lak",	"lbe"),
    ("Lezghian",	"lez"),
    ("Lithuanian",	"lt"),
    ("Latvian",	"lv"),
    ("Magahi",	"mah"),
    ("Maithili",	"mai"),
    ("Maori",	"mi"),
    ("Mongolian",	"mn"),
    ("Marathi",	"mr"),
    ("Malay",	"ms"),
    ("Maltese",	"mt"),
    ("Nepali",	"ne"),
    ("Newari",	"new"),
    ("Dutch",	"nl"),
    ("Norwegian",	"no"),
    ("Occitan",	"oc"),
    ("Pali",	"pi"),
    ("Polish",	"pl"),
    ("Portuguese",	"pt"),
    ("Romanian",	"ro"),
    ("Russian",	"ru"),
    ("Serbian (cyrillic)"	"rs_cyrillic"),
    ("Serbian, (latin)"	"rs_latin"),
    ("Nagpuri",	"sck"),
    ("Slovak",	"sk"),
    ("Slovenian",	"sl"),
    ("Albanian",	"sq"),
    ("Swedish",	"sv"),
    ("Swahili",	"sw"),
    ("Tamil",	"ta"),
    ("Tabassaran",	"tab"),
    ("Telugu",	"te"),
    ("Thai",	"th"),
    ("Tajik",	"tjk"),
    ("Tagalog",	"tl"),
    ("Turkish",	"tr"),
    ("Uyghur",	"ug"),
    ("Ukranian",	"uk"),
    ("Urdu",	"ur"),
    ("Uzbek",	"uz"),
    ("Vietnamese",	"vi"),
    ]


class OcrModel(c.Module):

    whitelist = ['readtext', 'detect', 'gradio']

    def __init__(self, api_key: str = None, cache_key: bool = True):
        config = self.set_config(kwargs=locals())

    def set_api_key(self, api_key: str, cache: bool = True):
        if api_key == None:
            api_key = self.get_api_key()

        self.api_key = api_key
        if cache:
            self.add_api_key(api_key)

        assert isinstance(api_key, str)

    def readtext(self, image, lang_list = ['en'], decoder = 'greedy', beamWidth= 5, batch_size = 1,
                 workers = 0, allowlist = None, blocklist = None, detail = 1,
                 rotation_info = None, paragraph = False, min_size = 20,
                 ):
        """
        readtext method
        Main method for Reader object. There are 4 groups of parameter: General, Contrast, Text Detection and Bounding Box Merging.

        Parameters:
            image (string, numpy array, byte) - Input image
            decoder (string, default = 'greedy') - options are 'greedy', 'beamsearch' and 'wordbeamsearch'.
            beamWidth (int, default = 5) - How many beam to keep when decoder = 'beamsearch' or 'wordbeamsearch'
            batch_size (int, default = 1) - batch_size>1 will make EasyOCR faster but use more memory
            workers (int, default = 0) - Number thread used in of dataloader
            allowlist (string) - Force EasyOCR to recognize only subset of characters. Useful for specific problem (E.g. license plate, etc.)
            blocklist (string) - Block subset of character. This argument will be ignored if allowlist is given.
            detail (int, default = 1) - Set this to 0 for simple output
            paragraph (bool, default = False) - Combine result into paragraph
            min_size (int, default = 10) - Filter text box smaller than minimum value in pixel
            rotation_info (list, default = None) - Allow EasyOCR to rotate each text box and return the one with the best confident score. Eligible values are 90, 180 and 270. For example, try [90, 180 ,270] for all possible text orientations.

        Return list of results
        """
        reader = easyocr.Reader(lang_list)
        result = reader.readtext(image, decoder, beamWidth, batch_size,
                 workers, allowlist, blocklist, detail,
                 rotation_info, paragraph, min_size,
                 )

        return result

    def detect(self, image, lang_list = ['en'], min_size = 20, text_threshold = 0.7, low_text = 0.4,\
               link_threshold = 0.4,canvas_size = 2560, mag_ratio = 1.,\
               slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
               width_ths = 0.5, add_margin = 0.1, reformat=True, optimal_num_chars=None,
               threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
               ):
        """
        detect method
        Method for detecting text boxes.

        Parameters
        image (string, numpy array, byte) - Input image
        min_size (int, default = 10) - Filter text box smaller than minimum value in pixel
        text_threshold (float, default = 0.7) - Text confidence threshold
        low_text (float, default = 0.4) - Text low-bound score
        link_threshold (float, default = 0.4) - Link confidence threshold
        canvas_size (int, default = 2560) - Maximum image size. Image bigger than this value will be resized down.
        mag_ratio (float, default = 1) - Image magnification ratio
        slope_ths (float, default = 0.1) - Maximum slope (delta y/delta x) to considered merging. Low value means tiled boxes will not be merged.
        ycenter_ths (float, default = 0.5) - Maximum shift in y direction. Boxes with different level should not be merged.
        height_ths (float, default = 0.5) - Maximum different in box height. Boxes with very different text size should not be merged.
        width_ths (float, default = 0.5) - Maximum horizontal distance to merge boxes.
        add_margin (float, default = 0.1) - Extend bounding boxes in all direction by certain value. This is important for language with complex script (E.g. Thai).
        optimal_num_chars (int, default = None) - If specified, bounding boxes with estimated number of characters near this value are returned first.
        Return horizontal_list, free_list - horizontal_list is a list of regtangular text boxes. The format is [x_min, x_max, y_min, y_max]. free_list is a list of free-form text boxes. The format is [[x1,y1],[x2,y2],[x3,y3],[x4,y4]].
        """
        reader = easyocr.Reader(lang_list)
        result = reader.detect(image, min_size = min_size, text_threshold = text_threshold, low_text = low_text,\
               link_threshold = link_threshold, canvas_size = canvas_size, mag_ratio = mag_ratio,\
               slope_ths = slope_ths, ycenter_ths = ycenter_ths, height_ths = height_ths,\
               width_ths = width_ths, add_margin = add_margin, reformat=reformat, optimal_num_chars=optimal_num_chars,
               threshold = threshold, bbox_min_score = bbox_min_score, bbox_min_size = bbox_min_size, max_candidates = max_candidates,
               )

        return result

    def recognize(self, image, lang_list = ['en'], horizontal_list=None, free_list=None,\
                  decoder = 'greedy', beamWidth= 5, batch_size = 1,\
                  workers = 0, allowlist = None, blocklist = None, detail = 1,\
                  rotation_info = None,paragraph = False,\
                  contrast_ths = 0.1,adjust_contrast = 0.5, filter_ths = 0.003,\
                  y_ths = 0.5, x_ths = 1.0, reformat=True, output_format='standard'
               ):
        """
        recognize method
        Method for recognizing characters from text boxes. If horizontal_list and free_list are not given. It will treat the whole image as one text box.

        Parameters
        image (string, numpy array, byte) - Input image
        horizontal_list (list, default=None) - see format from output of detect method
        free_list (list, default=None) - see format from output of detect method
        decoder (string, default = 'greedy') - options are 'greedy', 'beamsearch' and 'wordbeamsearch'.
        beamWidth (int, default = 5) - How many beam to keep when decoder = 'beamsearch' or 'wordbeamsearch'
        batch_size (int, default = 1) - batch_size>1 will make EasyOCR faster but use more memory
        workers (int, default = 0) - Number thread used in of dataloader
        allowlist (string) - Force EasyOCR to recognize only subset of characters. Useful for specific problem (E.g. license plate, etc.)
        blocklist (string) - Block subset of character. This argument will be ignored if allowlist is given.
        detail (int, default = 1) - Set this to 0 for simple output
        paragraph (bool, default = False) - Combine result into paragraph
        contrast_ths (float, default = 0.1) - Text box with contrast lower than this value will be passed into model 2 times. First is with original image and second with contrast adjusted to 'adjust_contrast' value. The one with more confident level will be returned as a result.
        adjust_contrast (float, default = 0.5) - target contrast level for low contrast text box
        Return list of results
        """
        reader = easyocr.Reader(lang_list)
        result = reader.recognize(image, horizontal_list=horizontal_list, free_list=free_list,\
                  decoder = decoder, beamWidth= beamWidth, batch_size = batch_size,\
                  workers = 0, allowlist = allowlist, blocklist = blocklist, detail = detail,\
                  rotation_info = rotation_info,paragraph = paragraph,\
                  contrast_ths = contrast_ths,adjust_contrast = adjust_contrast, filter_ths = filter_ths,\
                  y_ths = y_ths, x_ths = x_ths, reformat=reformat, output_format=output_format,
               )

        return result

    def readtextForGradio(self, img_path, lang_list = ['en']):
        image = cv2.imread(img_path)
        results = self.readtext(image, lang_list)
        for result in results:
            (pos, text, acc) = result
            cv2.rectangle(
                image, (int(pos[0][0]), int(pos[0][1])), (int(pos[2][0]), int(pos[2][1])), (255, 255, 255), 2
            )
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), results
            

    def gradio(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    img_path = gr.Image(label="Image", type="filepath")
                    lang_list = gr.Dropdown(
                        languages, label="Language Codes", info="Supported Languages", value=languages[0], multiselect=True)
                    readtext_but = gr.Button('Read Text')
                with gr.Column():
                    output_img = gr.Image(label="Output")
                    output_text = gr.Textbox(label="Output")
                readtext_but.click(fn=self.readtextForGradio, inputs=[img_path, lang_list], outputs=[output_img, output_text])
        demo.launch(quiet=True, share=True)
