from pathlib import Path
from time import perf_counter
import math

import pymupdf
import torch
from PIL import Image
from transformers import AutoImageProcessor, TableTransformerForObjectDetection

MODEL_NAME = 'microsoft/table-transformer-detection'
MODEL_REVISION = '2357cbe2b5a5d1c03e54f32764f06058933b65ab'


def load_table_detector(model_name=MODEL_NAME, revision=MODEL_REVISION, device='auto', cache_dir=None, local_files_only=False):
    """지정한 커밋의 사전학습 탐지 모델과 전처리기를 로드함."""
    if not revision or len(revision) != 40:
        raise ValueError('재현 가능한 40자리 모델 커밋 지정 필요함')
    resolved_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device == 'auto' else torch.device(device)
    if resolved_device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('현재 PyTorch 환경에서 CUDA 사용 불가함')
    options = {'revision': revision, 'cache_dir': str(cache_dir) if cache_dir else None, 'local_files_only': local_files_only}
    processor = AutoImageProcessor.from_pretrained(model_name, backend="pil", **options)
    model, loading_info = TableTransformerForObjectDetection.from_pretrained(
        model_name, use_safetensors=True, output_loading_info=True, **options)
    if loading_info.get('missing_keys') or loading_info.get('mismatched_keys') or loading_info.get('error_msgs'):
        raise RuntimeError(f'가중치 로드 완전성 확인 실패함: {loading_info}')
    model.to(resolved_device)
    model.eval()
    model.requires_grad_(False)
    return processor, model, resolved_device, loading_info


def render_pdf_page(pdf_path, page_number, dpi=150):
    """PDF의 지정 페이지를 비율 유지 RGB 이미지로 렌더링함."""
    path = Path(pdf_path)
    if not path.is_file():
        raise FileNotFoundError(path)
    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError('PDF 페이지 번호는 1 이상의 정수여야 함')
    if dpi <= 0:
        raise ValueError('DPI는 양수여야 함')
    with pymupdf.open(path) as document:
        if document.needs_pass:
            raise ValueError('암호화된 PDF는 별도 처리 필요함')
        if page_number > len(document):
            raise ValueError(f'PDF 페이지 범위 초과함: {page_number} / {len(document)}')
        page = document[page_number - 1]
        rect = page.rect
        pixmap = page.get_pixmap(dpi=dpi, alpha=False, colorspace=pymupdf.csRGB)
        image = Image.frombytes('RGB', (pixmap.width, pixmap.height), pixmap.samples)
        metadata = {'page':page_number, 'dpi':dpi, 'image_width':image.width, 'image_height':image.height,
                    'pdf_rect':list(rect), 'pdf_width_points':rect.width, 'pdf_height_points':rect.height,
                    'rotation':page.rotation, 'coordinate_system':'회전 반영 page.rect 기준, 좌상단 원점, PDF point 단위임'}
    return image, metadata


def image_bbox_to_pdf(bbox, page_metadata):
    """이미지 픽셀 좌표를 회전 반영 PDF 페이지 좌표로 변환함."""
    if len(bbox) != 4 or not all(math.isfinite(float(value)) for value in bbox):
        raise ValueError('Bounding Box는 유한한 숫자 4개여야 함')
    x0, y0, x1, y1 = map(float, bbox)
    if x1 <= x0 or y1 <= y0:
        raise ValueError('Bounding Box의 가로·세로 길이가 양수여야 함')
    width, height = page_metadata['image_width'], page_metadata['image_height']
    if width <= 0 or height <= 0:
        raise ValueError('이미지 크기가 양수여야 함')
    left, top, right, bottom = page_metadata['pdf_rect']
    return [left+x0*(right-left)/width, top+y0*(bottom-top)/height,
            left+x1*(right-left)/width, top+y1*(bottom-top)/height]


def detect_tables(image, processor, model, device, threshold=0.5):
    """이미지에서 임계값 이상 표 영역을 탐지하고 실제 시간을 반환함."""
    return _predict_table_objects(image, processor, model, device, threshold, {'table', 'table rotated'})


def _predict_table_objects(image, processor, model, device, threshold, allowed_labels):
    """공통 전처리·추론·좌표 복원 후 지원하는 클래스만 반환함."""
    if not 0 <= threshold <= 1:
        raise ValueError('신뢰도 임계값은 0~1 범위여야 함')
    if image.mode != 'RGB':
        image = image.convert('RGB')
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    started = perf_counter()
    inputs = processor(images=image, return_tensors='pt').to(device)
    preprocessed = perf_counter()
    with torch.inference_mode():
        outputs = model(**inputs)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)
    inferred = perf_counter()
    target_sizes = torch.tensor([[image.height, image.width]], device=device)
    result = processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)[0]
    detections = []
    for score, label, box in zip(result['scores'], result['labels'], result['boxes']):
        label_id = int(label.item())
        label_name = model.config.id2label[label_id]
        if label_name not in allowed_labels:
            continue
        bbox = box.detach().cpu().tolist()
        if not all(math.isfinite(value) for value in bbox):
            raise ValueError('모델 출력 좌표가 유한하지 않음')
        detections.append({'detection_id':len(detections)+1, 'label':label_name, 'confidence':float(score.item()), 'bbox':bbox})
    finished = perf_counter()
    timing = {'processing_seconds':finished-started, 'preprocessing_seconds':preprocessed-started,
              'inference_seconds':inferred-preprocessed, 'postprocessing_seconds':finished-inferred,
              'model_input_height':int(inputs['pixel_values'].shape[-2]), 'model_input_width':int(inputs['pixel_values'].shape[-1])}
    return detections, timing


def detection_rows(document_id, model_name, detections, page_metadata, timing, model_revision=MODEL_REVISION):
    """탐지 영역별 CSV 행을 만들고 무탐지 페이지도 한 행으로 보존함."""
    import json
    common = {'document_id':document_id, 'page':page_metadata['page'], 'model_name':model_name,
              'image_width':page_metadata['image_width'], 'image_height':page_metadata['image_height'],
              'detection_count':len(detections), 'processing_seconds':timing['processing_seconds'],
              'model_revision':model_revision, 'dpi':page_metadata['dpi'], 'rotation':page_metadata['rotation'],
              'model_input_width':timing['model_input_width'], 'model_input_height':timing['model_input_height'],
              'preprocessing_seconds':timing['preprocessing_seconds'], 'inference_seconds':timing['inference_seconds'],
              'postprocessing_seconds':timing['postprocessing_seconds']}
    if not detections:
        return [{**common, 'detection_id':0, 'label':'', 'confidence':None, 'bbox':'', 'bbox_pdf':''}]
    return [{**common, 'detection_id':item['detection_id'], 'label':item['label'], 'confidence':item['confidence'],
             'bbox':json.dumps(item['bbox']), 'bbox_pdf':json.dumps(image_bbox_to_pdf(item['bbox'],page_metadata))}
            for item in detections]

def expand_table_bbox(bbox, image_width, image_height, padding_ratio):
    """상자 너비·높이의 비율만큼 각 방향을 확장하고 이미지 경계로 제한함."""
    values = [*bbox, image_width, image_height, padding_ratio]
    if len(bbox) != 4 or not all(math.isfinite(value) for value in values):
        raise ValueError('좌표·크기·여백은 유한해야 함')
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0 or image_width <= 0 or image_height <= 0 or padding_ratio < 0:
        raise ValueError('상자·이미지 크기는 양수, 여백 비율은 0 이상이어야 함')
    dx = (x1 - x0) * padding_ratio
    dy = (y1 - y0) * padding_ratio
    result = [max(0., x0 - dx), max(0., y0 - dy),
              min(float(image_width), x1 + dx), min(float(image_height), y1 + dy)]
    if result[0] >= result[2] or result[1] >= result[3]:
        raise ValueError('확장 상자가 이미지와 교차하지 않음')
    return result


def extract_pdf_page_text(pdf_path, page_number, bbox_pdf=None):
    """페이지 원문과 선택 영역 원문을 분리하고 PyMuPDF 단어 좌표를 유지함."""
    import pdfplumber

    if not isinstance(page_number, int) or isinstance(page_number, bool) or page_number < 1:
        raise ValueError('PDF 페이지 번호는 1 이상의 정수여야 함')
    with pymupdf.open(pdf_path) as document:
        if document.needs_pass or page_number > len(document):
            raise ValueError('암호화된 PDF 또는 범위 밖 페이지임')
        page = document[page_number - 1]
        if page.rotation or page.cropbox != page.mediabox:
            raise ValueError('회전 또는 다른 CropBox의 도구 간 좌표 일치 검증 필요함')
        rect = page.rect
        full_text = page.get_text('text', sort=True)
        words = page.get_text('words', sort=True)
        region = pymupdf.Rect(bbox_pdf) if bbox_pdf is not None else None
        if region is not None and (region.is_empty or not rect.contains(region)):
            raise ValueError('영역 좌표는 페이지 안의 양수 면적이어야 함')
        clipped_text = page.get_text('text', clip=region, sort=True) if region is not None else None
    with pdfplumber.open(pdf_path) as document:
        page = document.pages[page_number - 1]
        if not all(math.isclose(a, b, abs_tol=1e-3) for a,b in zip(rect,page.bbox)):
            raise ValueError('PyMuPDF와 pdfplumber 페이지 좌표가 일치하지 않음')
        plumber_text = page.extract_text() or ''
        plumber_clip = (page.within_bbox(tuple(region)).extract_text() or '') if region is not None else None
    return {'page':page_number, 'page_bbox_pdf':list(rect),
            'bbox_pdf':list(region) if region is not None else None,
            'page_text_pymupdf':full_text, 'page_text_pdfplumber':plumber_text,
            'region_text_pymupdf':clipped_text, 'region_text_pdfplumber':plumber_clip,
            'page_words':words, 'region_words':[w for w in words if region.contains(pymupdf.Rect(w[:4]))]
                if region is not None else None}


STRUCTURE_MODEL_NAME = 'microsoft/table-transformer-structure-recognition'
STRUCTURE_MODEL_REVISION = 'f4d4bdc85c3fe4b1fa49658882a5d38bbdd0f343'
STRUCTURE_LABELS = frozenset({'table', 'table row', 'table column', 'table column header',
                             'table projected row header', 'table spanning cell'})


def load_structure_recognizer(device='auto', cache_dir=None, local_files_only=False):
    """고정 커밋의 표 구조 인식 모델을 평가 모드로 로드함."""
    processor, model, device, info = load_table_detector(
        STRUCTURE_MODEL_NAME, STRUCTURE_MODEL_REVISION, device, cache_dir, local_files_only)
    if set(model.config.id2label.values()) != STRUCTURE_LABELS:
        raise RuntimeError('구조 모델의 클래스 구성 불일치함')
    return processor, model, device, info


def crop_table_image(page_image, expanded_bbox):
    """페이지를 원래 픽셀 밀도로 자르고 반올림한 영역 좌표를 반환함."""
    x0,y0,x1,y1 = expand_table_bbox(expanded_bbox, page_image.width, page_image.height, 0)
    crop_box = (math.floor(x0),math.floor(y0),math.ceil(x1),math.ceil(y1))
    return page_image.crop(crop_box), crop_box


def structure_bbox_to_page(bbox, crop_box, page_metadata):
    """구조 모델의 로컬 좌표를 잘라낸 이미지에 제한하고 페이지 픽셀·PDF 좌표로 변환함."""
    left,top,right,bottom = crop_box
    if not (0 <= left < right <= page_metadata['image_width'] and
            0 <= top < bottom <= page_metadata['image_height']):
        raise ValueError('잘라낸 영역이 페이지 경계를 벗어남')
    clipped = expand_table_bbox(bbox,right-left,bottom-top,0)
    page_bbox = [clipped[0]+left,clipped[1]+top,clipped[2]+left,clipped[3]+top]
    return clipped, page_bbox, image_bbox_to_pdf(page_bbox,page_metadata)


def recognize_table_structure(crop_image, processor, model, device, threshold=0.5):
    """표 이미지에서 지원하는 구조 객체와 처리 시간을 반환함."""
    return _predict_table_objects(crop_image,processor,model,device,threshold,STRUCTURE_LABELS)

