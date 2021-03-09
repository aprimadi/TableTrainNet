import pytesseract
import tensorflow as tf
import numpy as np
import cv2
import copy
import requests
import io
import json
import re
from PIL import Image
from enum import Enum
from object_detection.utils import visualization_utils as vis_util

API_KEY = '3df1501bb688957'

def ocr_space_file(filename, file, overlay=False, api_key=API_KEY, language='eng'):
    """ OCR.space API request with local file.
        Python3.5 - not tested on 2.7
    :param filename: Your file path & name.
    :param overlay: Is OCR.space overlay required in your response.
                    Defaults to False.
    :param api_key: OCR.space API key.
    :param language: Language code to be used in OCR.
                    List of available language codes can be found on https://ocr.space/OCRAPI
                    Defaults to 'en'.
    :return: Result in JSON format.
    """

    payload = {'isOverlayRequired': overlay,
               'apikey': api_key,
               'language': language,
               }
    r = requests.post('https://api.ocr.space/parse/image',
                      files={filename: file},
                      data=payload)
    return r.content.decode()

def preprocess_image(img):
    # Grayscale the image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Perform histogram equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img = clahe.apply(img)

    (im_width, im_height) = img.shape
    img = img.reshape((im_width, im_height, 1)).astype(np.uint8)
    img = np.repeat(img, 3, axis=-1)

    return img

def is_vertically_overlapped(box_a, box_b):
    return (
        box_a['y_min'] <= box_b['y_min'] <= box_a['y_max'] or
        box_a['y_min'] <= box_b['y_max'] <= box_a['y_max'] or
        box_b['y_min'] <= box_a['y_min'] <= box_b['y_max'] or
        box_b['y_min'] <= box_a['y_max'] <= box_b['y_max']
    )

def merge_vertically_overlapping_boxes(boxes):
    merged_boxes = [boxes[0]]
    any_box_merged = False
    for box in boxes[1:]:
        merged = False
        for m_box in merged_boxes:
            coord_m_box = {
                'y_min': m_box[0],
                'x_min': m_box[1],
                'y_max': m_box[2],
                'x_max': m_box[3],
            }
            coord_box = {
                'y_min': box[0],
                'x_min': box[1],
                'y_max': box[2],
                'x_max': box[3],
            }
            if is_vertically_overlapped(coord_m_box, coord_box):
                merged = True
                any_box_merged = True
                if m_box[0] > box[0]:
                    m_box[0] = box[0]
                if m_box[2] < box[2]:
                    m_box[2] = box[2]
        if not merged:
            merged_boxes.append(box)
    if any_box_merged:
        return merge_vertically_overlapping_boxes(merged_boxes)
    else:
        return merged_boxes

def keep_best_boxes(boxes, scores, max_num_boxes=5, min_score=0.8):
    """
    return a list of the max_num_boxes not overlapping boxes found in inference
    
    boxes are: box[0]=ymin, box[1]=xmin, box[2]=ymax, box[3]=xmax

    :param boxes: list of boxes found in inference
    :param scores: likelihood of the boxes
    :param max_num_boxes: max num of boxes to be saved
    :param min_score: min box score to check
    :return: list of the best boxes
    """
    kept_boxes = []
    num_boxes = 0
    if scores[0] > min_score:
        kept_boxes.append(boxes[0]) # always keep the first box, which is the best one
        num_boxes += 1
        for b in boxes[1:]:
            if num_boxes < max_num_boxes and scores[num_boxes] > min_score:
                kept_boxes.append(b)
                num_boxes += 1
            else:
                break
        # keep the overlapping boxes
        kept_boxes = merge_vertically_overlapping_boxes(kept_boxes)
    else:
        kept_boxes = []

    return kept_boxes

def wrap_frozen_graph(graph_def, inputs, outputs):
    def _imports_graph_def():
        tf.compat.v1.import_graph_def(graph_def, name="")
    wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
    import_graph = wrapped_import.graph

    return wrapped_import.prune(
        tf.nest.map_structure(import_graph.as_graph_element, inputs),
        tf.nest.map_structure(import_graph.as_graph_element, outputs))

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

class Range:
    """
    Represents a range of number from [low, high] (inclusive).
    """
    
    def __init__(self, low, high):
        assert(low <= high)
        self.low = low
        self.high = high

    def __repr__(self):
        return str((self.low, self.high))

    def is_intersecting(self, rng):
        l1, r1 = self.low, self.high
        l2, r2 = rng.low, rng.high
        return (
            l2 <= l1 <= r2 or
            l2 <= r1 <= r2 or
            l1 <= l2 <= r1 or
            l1 <= r2 <= r1
        )

    def union(self, rng):
        """Returns a new range that is a union between this range and the given range."""
        l1, r1 = self.low, self.high
        l2, r2 = rng.low, rng.high
        return Range(min(l1, l2), max(r1, r2))

class BoundingBox:
    def __init__(self, top, left, width, height):
        self.top = top
        self.left = left
        self.width = width
        self.height = height

    def as_tuple(self):
        return (self.top, self.left, self.width, self.height)

    def is_enclosing(self, box):
        """Check if this box totally enclose the given box"""
        (t1, l1, w1, h1) = self.as_tuple()
        (t2, l2, w2, h2) = box.as_tuple()
        b1, r1 = t1 + h1, l1 + w1
        b2, r2 = t2 + h2, l2 + w2
        return (t1 <= t2 and l1 <= l2 and b1 >= b2 and r1 >= r2)

class DataType(Enum):
    NullType = 1
    YearType = 2
    NumberType = 3
    StringType = 4

class TypeInference:
    def infer(self, field):
        if field is None:
            return DataType.NullType
        else:
            return self._try_year_type(field)

    def cast(self, field, field_type):
        if field_type == DataType.NullType:
            return None
        elif field_type == DataType.YearType:
            return int(field)
        elif field_type == DataType.NumberType:
            if isinstance(field, str):
                field = self._sanitize_number_str(field)
            try:
                return int(field)
            except Exception:
                return float(field)
        elif field_type == DataType.StringType:
            return str(field)
        else:
            raise AssertionError("Unsupported field_type: " + str(field_type))

    def _try_year_type(self, field):
        try:
            value = int(field)
            if value >= 1970 and value <= 2100:
                return DataType.YearType
            else:
                return DataType.NumberType
        except Exception:
            return self._try_integral_type(field)

    def _try_integral_type(self, field):
        try:
            if isinstance(field, str):
                _field = self._sanitize_number_str(field)
            int(_field)
            return DataType.NumberType
        except Exception:
            return self._try_floating_type(field)
    
    def _try_floating_type(self, field):
        try:
            if isinstance(field, str):
                _field = self._sanitize_number_str(field)
            float(_field)
            return DataType.NumberType
        except Exception:
            return DataType.StringType

    def _sanitize_number_str(self, input_str):
        return re.sub(r"(\s|,)", "", input_str)

class TableExtractor:
    CONFIDENCE_THRESHOLD = 60

    class OcrResult:
        def __init__(self, top, left, width, height, text):
            self.bounding_box = BoundingBox(top, left, width, height)
            self.text = text

        def __repr__(self):
            return str(self.bounding_box.as_tuple() + (self.text,))

    def extract_table(self, image):
        ocr_results = self._extract_words(image)
        avg_text_height = self._calculate_avg_text_height(ocr_results)
        tops = self._populate_tops(ocr_results, avg_text_height)
        column_ranges = self._populate_column_ranges(ocr_results)

        rows = len(tops)
        cols = len(column_ranges)
        table = [[None for c in range(cols)] for r in range(rows)]
        for res in ocr_results:
            tup = self._cell_index(res.bounding_box, tops, avg_text_height, column_ranges)
            if tup is not None:
                (row, col) = tup
                table[row][col] = res.text

        table = self._reduce_table(table)
        table = self._combine_multiline_rows(table)
        table = self._cast_table(table)

        return table

    def _reduce_table(self, table):
        """Remove blank rows and columns from table"""
        new_table = copy.deepcopy(table)
        if len(new_table) == 0:
            return new_table

        rows = len(new_table)
        cols = len(new_table[0])
        # Clean up empty rows
        for row in range(rows):
            empty = True
            for cell in new_table[row]:
                if cell is not None:
                    empty = False
                    break
            if empty:
                del(new_table[row])
        # Clean up empty cols
        for col in range(cols):
            empty = True
            for row in range(rows):
                if new_table[row][col] is not None:
                    empty = False
                    break
            if empty:
                for row in range(rows):
                    del(new_table[row][col])
        return new_table

    def _combine_multiline_rows(self, table):
        """Try combining multiline rows"""
        row_types = self._table_row_types(table)
        most_used_row_type = self._most_used_row_type(row_types)

        # Combine multiline row
        idx = 0
        header = True
        res_table = []
        row = None
        while idx < len(table):
            row_type = row_types[idx]

            if header:
                # On header row, always append row until we find a non-header row
                if str(row_type) == most_used_row_type:
                    # Non header row found, store the row
                    header = False
                    row = table[idx]
                else:
                    res_table.append(table[idx])
            else:
                # Non-header, always store the row first until we find a start 
                # of a line

                # Try to combine it with an existing row
                if row is not None:
                    combined_row = self._try_combine_row(row, table[idx])
                    if combined_row is not None:
                        row = combined_row
                    else:
                        # This marks the beginning of a line, append row 
                        # and store this row to be processed later
                        res_table.append(row)
                        row = table[idx]
                else:
                    if str(row_type) == most_used_row_type:
                        # This marks the start of a line
                        if row is not None:
                            res_table.append(row)
                    row = table[idx]

            idx += 1
        if row is not None:
            res_table.append(row)

        return res_table

    def _cast_table(self, table):
        type_inference = TypeInference()
        res_table = copy.deepcopy(table)
        for i in range(len(table)):
            for j in range(len(table[i])):
                cell = table[i][j]
                ctype = type_inference.infer(cell)
                print(ctype)
                cell = type_inference.cast(cell, ctype)
                res_table[i][j] = cell
        return res_table

    def _table_row_types(self, table):
        """Return the row types for each row of the table."""
        row_types = []
        for row in table:
            row_types.append(self._infer_row_type(row))
        return row_types

    def _most_used_row_type(self, row_types):
        """Find out most used row type"""
        row_type_count = {}   # Map from rowType -> count
        for row_type in row_types:
            row_type_count[str(row_type)] = row_type_count.get(str(row_type), 0) + 1
        most_used = None
        most_used_count = 0
        for row_type in row_type_count:
            if row_type_count[str(row_type)] > most_used_count:
                most_used = row_type
                most_used_count = row_type_count[str(row_type)]
        return most_used

    def _try_combine_row(self, row1, row2):
        """Try combining two adjacent rows in a table.

        Returns a new row that is the result of combining rows or None if they cannot be combined.
        """
        row_type_1 = self._infer_row_type(row1)
        row_type_2 = self._infer_row_type(row2)
        if self._compatible_row_type(row_type_1, row_type_2):
            new_row = []
            ntypes = len(row_type_1)
            for idx in range(ntypes):
                ctype1 = row_type_1[idx]
                ctype2 = row_type_2[idx]
                if ctype1 == DataType.NullType:
                    new_row.append(row2[idx])
                elif ctype1 == DataType.StringType and ctype2 == DataType.StringType:
                    new_row.append(row1[idx] + " " + row2[idx])
                else:
                    raise AssertionError(f"row type combination is valid when it shouldn't: {ctype1}, {ctype2}")
            return new_row
        else:
            return None

    def _compatible_row_type(self, row_type_1, row_type_2):
        """Check if row_type_2 is compatible and can be combined into row_type_1"""

        ntypes = len(row_type_2)
        for idx in range(ntypes):
            ctype1 = row_type_1[idx]
            ctype2 = row_type_2[idx]
            # ctype2 is compatible with ctype1 if
            # ctype2 is any type and ctype1 is NullType
            # ctype2 is StringType and ctype1 is StringType
            if ctype1 not in [DataType.NullType, DataType.StringType]:
                return False
            elif ctype1 == DataType.StringType and ctype2 != DataType.StringType:
                return False
        return True

    def _infer_row_type(self, row):
        ctypes = []
        type_inference = TypeInference()
        for cell in row:
            ctypes.append(type_inference.infer(cell))
        return ctypes

    def _extract_words(self, image):
        b = io.BytesIO()
        image.save(b, "PNG")
        b.seek(0)
        data = json.loads(ocr_space_file("cropped.png", b, True))

        ocr_results = []
        for parsed_result in data['ParsedResults']:
            for line in parsed_result['TextOverlay']['Lines']:
                dim_set = False
                t, l, w, h = None, None, None, None
                text = line['LineText']
                for word in line['Words']:
                    if not dim_set:
                        dim_set = True
                        (t, l, w, h) = (word['Top'], word['Left'], word['Width'], word['Height'])
                    else:
                        t = min(t, word['Top'])
                        l = min(l, word['Left'])
                        w = max(w, word['Width'])
                        h = max(h, word['Height'])
                if dim_set:
                    ocr_results.append(TableExtractor.OcrResult(t, l, w, h, text))
        return ocr_results

    def _extract_words_tesseract(self, image):
        d = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT, config="--psm 13")
        n_boxes = len(d['text'])
        ocr_results = []
        for i in range(n_boxes):
            (t, l, w, h) = (d['top'][i], d['left'][i], d['width'][i], d['height'][i])
            if int(d['conf'][i]) > TableExtractor.CONFIDENCE_THRESHOLD:
                text = d['text'][i]
                ocr_results.append(TableExtractor.OcrResult(t, l, w, h, text))
        return ocr_results

    def _calculate_avg_text_height(self, ocr_results):
        if len(ocr_results) > 0:
            total = 0.0
            count = 0
            for res in ocr_results:
                h = res.bounding_box.height
                total += h
                count += 1
            return total / count
        else:
            return 10

    def _populate_tops(self, ocr_results, avg_text_height):
        """Group text by top, this in effect group text by row"""
        tops = []
        for res in ocr_results:
            t = res.bounding_box.top
            idx = 0
            found = False
            while idx < len(tops):
                top = tops[idx]
                lo = top - avg_text_height / 2
                hi = top + avg_text_height / 2
                if t < lo:
                    break
                elif lo <= t and t <= hi:
                    found = True
                    break
                idx += 1
            if not found:
                tops.insert(idx, t)
        return tops

    def _populate_column_ranges(self, ocr_results):
        column_ranges = []
        for res in ocr_results:
            (_, left, width, _) = res.bounding_box.as_tuple()
            right = left + width
            idx = 0
            found = False
            while idx < len(column_ranges):
                rng = column_ranges[idx]
                if right < rng.low:
                    break
                elif rng.is_intersecting(Range(left, right)):
                    found = True
                    break
                idx += 1
            if not found:
                column_ranges.insert(idx, Range(left, right))
            else:
                column_ranges[idx] = column_ranges[idx].union(Range(left, right))
        return column_ranges

    def _cell_index(self, box, tops, avg_text_height, column_ranges):
        """
        Returns the row and column index for a given bounding box.

        Returns None if the row and column cannot be determined.
        """
        t, l, w, _ = box.as_tuple()
        row = -1
        col = -1
        for i in range(len(tops)):
            top = tops[i]
            if top - avg_text_height / 2 <= t <= top + avg_text_height / 2:
                row = i
                break
        for i in range(len(column_ranges)):
            if column_ranges[i].is_intersecting(Range(l, l+w)):
                col = i
                break
        if row == -1 or col == -1:
            return None
        else:
            return (row, col)

def predict():
    MIN_SCORE = 0.2
    MAX_NUM_BOXES = 20
    test_path = 'test/test8.bmp'
    # model_path = 'trained_models/model__rcnn_inception_adam_1/frozen/frozen_inference_graph.pb'
    model_path = 'trained_models/model__rcnn_inception_momentum_optimizer_1batch/frozen/frozen_inference_graph.pb'

    detection_graph = tf.Graph()
    with detection_graph.as_default():
        # other TF commands
        od_graph_def = tf.compat.v1.GraphDef()
        with tf.compat.v1.gfile.GFile(model_path, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    # DETECTION: THE NICE PART

    with detection_graph.as_default():
        with tf.compat.v1.Session(graph=detection_graph) as sess:
            image = Image.open(test_path)
            image.convert()
            # the array based representation of the image will be used later in order to prepare the
            # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)

            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            # print('scores', scores[0])  # scores[0][0]=score of first box

            # we merge the best vertical overlapping boxes
            best_boxes = keep_best_boxes(
                boxes=boxes[0],
                scores=scores[0],
                max_num_boxes=MAX_NUM_BOXES,
                min_score=MIN_SCORE
            )

            return best_boxes

def predict_new():
    MIN_SCORE = 0.2
    MAX_NUM_BOXES = 20
    test_path = 'test/test8.PNG'
    # model_path = 'trained_models/model__rcnn_inception_adam_1/frozen/frozen_inference_graph.pb'
    model_path = 'trained_models/model__rcnn_inception_momentum_optimizer_1batch/frozen/frozen_inference_graph.pb'

    graph_def = tf.compat.v1.GraphDef()
    loaded = graph_def.ParseFromString(open(model_path, 'rb').read())

    inception_func = wrap_frozen_graph(graph_def, 
        inputs='image_tensor:0', 
        outputs=['detection_boxes:0', 'detection_scores:0', 'detection_classes:0', 'num_detections:0'])

    image = cv2.imread(test_path)
    image = preprocess_image(image)

    # image = Image.open(test_path)
    # image.convert()

    # image = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image, axis=0)

    boxes, scores, classes, num_detections = inception_func(tf.convert_to_tensor(image_np_expanded))
    best_boxes = keep_best_boxes(
        boxes=boxes[0], 
        scores=scores[0].numpy(), 
        max_num_boxes=MAX_NUM_BOXES, 
        min_score=MIN_SCORE
    )

if __name__ == '__main__':
    DEBUG = False

    test_path = 'test/test8.PNG'

    best_boxes = predict()

    pil_image = Image.open(test_path)
    pil_image.convert()
    for box in best_boxes:
        top, left, bottom, right = box[0], box[1], box[2], box[3]
        vis_util.draw_bounding_box_on_image(
            pil_image,
            top,
            0,
            bottom,
            1,
            color='red',
            thickness=4
        )
    (im_width, im_height) = pil_image.size
    pil_image_np = np.array(pil_image.getdata()).reshape((im_height, im_width, 4)).astype(np.uint8)
    if DEBUG:
        cv2.imshow('image', pil_image_np)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    extractor = TableExtractor()
    for box in best_boxes:
        pil_image = Image.open(test_path)
        pil_image.convert()
        (im_width, im_height) = pil_image.size

        top, left, bottom, right = (
            box[0] * im_height, box[1] * im_width, box[2] * im_height, box[3] * im_width
        )
        left = 0
        right = im_width
        cropped = pil_image.crop((left, top, right, bottom))
        (cr_width, cr_height) = cropped.size
        if DEBUG:
            cropped_image_np = np.array(cropped.getdata()).reshape((cr_height, cr_width, 4)).astype(np.uint8)
            cv2.imshow('image', cropped_image_np)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        table = extractor.extract_table(cropped)
        print(table)

