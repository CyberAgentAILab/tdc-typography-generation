import html
import math
import os
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import skia


def capitalize_text(text: str, capitalize: int) -> str:
    text_tmp = ""
    for character in text:
        if capitalize == 1:
            character = character.upper()
        text_tmp += character
    return text_tmp


def text_align(
    textAlign: int, left: float, center: float, right: float, text_alpha_width: float
) -> float:
    if textAlign == 1:
        x = center - text_alpha_width / 2.0
    elif textAlign == 3:
        x = right - text_alpha_width
    elif textAlign == 2:
        x = left
    return x


def get_text_location_info(font: skia.Font, text_tmp: str) -> Tuple:
    glyphs = font.textToGlyphs(text_tmp)
    positions = font.getPos(glyphs)
    rects = font.getBounds(glyphs)
    return glyphs, positions, rects


def compute_text_alpha_width(
    positions: List, rects: List, letterSpacing: float
) -> Union[float, Any]:
    twidth = positions[-1].x() + rects[-1].right()
    if letterSpacing is not None:
        twidth += letterSpacing * (len(rects) - 1)
    return twidth


def add_letter_margin(x: float, letterSpacing: float) -> float:
    if letterSpacing is not None:
        x = x + letterSpacing
    return x


def get_text_actual_height(
    font: skia.Font,
):
    ascender = -1 * font.getMetrics().fAscent
    descender = font.getMetrics().fDescent
    leading = font.getMetrics().fLeading
    text_height = ascender + descender + leading
    return text_height


def get_text_alpha(
    element_data: Any,
    text_index: int,
    texts: List,
    font: skia.Font,
    H: int,
    W: int,
    w_rate: float,
) -> np.array:
    center_y = element_data["text_center_y"][text_index] * H
    center_x = element_data["text_center_x"][text_index] * W
    text_width = get_text_actual_width(element_data, text_index, texts, font, w_rate)
    left = center_x - text_width / 2.0
    right = center_x + text_width / 2.0
    ascender = -1 * font.getMetrics().fAscent
    descender = font.getMetrics().fDescent
    leading = font.getMetrics().fLeading
    line_height_scale = element_data["line_height"][text_index]
    line_height = (ascender + descender + leading) * line_height_scale
    surface = skia.Surface(W, H)
    canvas = surface.getCanvas()
    fill_paint = skia.Paint(
        AntiAlias=True,
        Color=skia.ColorSetRGB(255, 0, 0),
        Style=skia.Paint.kFill_Style,
    )
    fill_paint.setBlendMode(skia.BlendMode.kSrcOver)
    y = center_y - line_height * len(texts) / 2.0
    for text in texts:
        text = html.unescape(text)
        text = capitalize_text(text, element_data["capitalize"][text_index])
        _, positions, rects = get_text_location_info(font, text)
        if len(positions) == 0:
            continue
        text_alpha_width = compute_text_alpha_width(
            positions, rects, element_data["letter_spacing"][text_index] * w_rate
        )
        # print(text,element_data["letter_spacing"][text_index],element_data["capitalize"][text_index])
        angle = float(element_data["angle"][text_index]) * 180 / math.pi
        x = text_align(
            element_data["text_align"][text_index],
            left,
            center_x,
            right,
            text_alpha_width,
        )
        canvas.rotate(angle, center_x, center_y)
        for i, character in enumerate(text):
            ydp = np.round(y + positions[i].y() + ascender)
            xdp = np.round(x + positions[i].x())
            textblob = skia.TextBlob(character, font)
            canvas.drawTextBlob(textblob, xdp, ydp, fill_paint)
            x = add_letter_margin(
                x, element_data["letter_spacing"][text_index] * w_rate
            )
        canvas.rotate(-1 * angle, center_x, y)
        y += line_height
    text_alpha = surface.makeImageSnapshot().toarray()[:, :, 3]
    text_alpha = text_alpha / 255.0
    text_alpha = np.tile(text_alpha[:, :, np.newaxis], (1, 1, 3))
    return np.minimum(text_alpha, np.zeros_like(text_alpha) + 1)


def get_text_actual_width(
    element_data: Any,
    text_index: int,
    texts: List,
    font: skia.Font,
    w_rate: float,
) -> np.array:
    text_alpha_width = 0.0
    for text in texts:
        text = html.unescape(text)
        text = capitalize_text(text, element_data["capitalize"][text_index])
        _, positions, rects = get_text_location_info(font, text)
        if len(positions) == 0:
            continue
        _text_alpha_width = compute_text_alpha_width(
            positions, rects, element_data["letter_spacing"][text_index] * w_rate
        )
        text_alpha_width = max(text_alpha_width, _text_alpha_width)
    return text_alpha_width


def font_name_fix(font_name: str) -> str:
    if font_name == "Exo_2":
        font_name = "Exo\_2"
    if font_name == "Press_Start_2P":
        font_name = "Press_Start\_2P"
    if font_name == "quattrocento":
        font_name = "Quattrocento"
    if font_name == "yellowtail":
        font_name = "Yellowtail"
    if font_name == "sunday":
        font_name = "Sunday"
    if font_name == "bebas_neue":
        font_name = "Bebas_Neue"
    if font_name == "Brusher":
        font_name = "Brusher_Regular"
    if font_name == "Amatic_Sc":
        font_name = "Amatic_SC"
    if font_name == "Pt_Sans":
        font_name = "PT_Sans"
    if font_name == "Old_Standard_Tt":
        font_name = "Old_Standard_TT"
    if font_name == "Eb_Garamond":
        font_name = "EB_Garamond"
    if font_name == "Gfs_Didot":
        font_name = "GFS_Didot"
    if font_name == "Im_Fell":
        font_name = "IM_Fell"
    if font_name == "Im_Fell_Dw_Pica_Sc":
        font_name = "IM_Fell_DW_Pica_SC"
    if font_name == "Marcellus_Sc":
        font_name = "Marcellus_SC"
    return font_name


def get_skia_font(
    font2ttf: dict,
    fontmgr: skia.FontMgr,
    element_data: Dict,
    targetid: int,
    font_name: str,
    scale_h: float,
    font_scale: float = 1.0,
) -> Tuple:
    font_name = font_name_fix(font_name)
    if font_name in font2ttf:
        ttf = font2ttf[font_name]
        ft = fontmgr.makeFromFile(ttf, 0)
        font = skia.Font(
            ft, element_data["font_size"][targetid] * scale_h, font_scale, 1e-20
        )

        return font, font_name
    else:
        ft = fontmgr.makeFromFile("", 0)
        font = skia.Font(
            ft, element_data["font_size"][targetid] * scale_h, font_scale, 1e-20
        )
        return None, None


def get_color_map(element_data: Any, targetid: int, H: int, W: int) -> np.array:
    B, G, R = element_data["color"][targetid]
    rgb_map = np.zeros((H, W, 3), dtype=np.uint8)
    rgb_map[:, :, 0] = B
    rgb_map[:, :, 1] = G
    rgb_map[:, :, 2] = R
    return rgb_map


def get_texts(element_data: Dict, target_id: int) -> List:
    text = html.unescape(element_data["text"][target_id])
    _texts = text.split(os.linesep)
    texts = []
    for t in _texts:
        if t == "":
            pass
        else:
            texts.append(t)
    return texts
