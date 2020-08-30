""" Script for api utils. """
from google.cloud import vision
from google.cloud.vision import types
import io

import pdb


def detect_label_numpy(image):
    client = vision.ImageAnnotatorClient()

    # Performs label detection on the image file
    response = client.label_detection(image=image)
    labels = response.label_annotations
    if len(labels) > 0:
        return labels[0]
    return None

def detect_label_file(path):
    client = vision.ImageAnnotatorClient()

    # The name of the image file to annotate
    with io.open(path, 'rb') as image_file:
        content = image_file.read()
    image = types.Image(content=content)
    # Performs label detection on the image file
    try:
        response = client.label_detection(image=image)
    except:
        return []
    labels = response.label_annotations
    return labels

def detect_objects_numpy(image):

    client = vision.ImageAnnotatorClient()
    objects = client.object_localization(
        image=image).localized_object_annotations


    if len(objects) > 0:
        return objects[0].name
    
    return None

def detect_objects_file(path):
    """Localize objects in the local image.

    Args:
    path: The path to the local file.
    """
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with open(path, 'rb') as image_file:
        content = image_file.read()
    image = vision.types.Image(content=content)
    try:
        objects = client.object_localization(
            image=image).localized_object_annotations
    except:
        objects = None

    '''
    print('Number of objects found: {}'.format(len(objects)))
    for object_ in objects:
        print('\n{} (confidence: {})'.format(object_.name, object_.score))
        print('Normalized bounding polygon vertices: ')
        for vertex in object_.bounding_poly.normalized_vertices:
            print(' - ({}, {})'.format(vertex.x, vertex.y))
    '''
    return objects

def detect_text_numpy(image):
    """Detects text in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    response = client.text_detection(image=image)
    texts = response.text_annotations
    if len(texts) > 0:
        return texts[0].description.strip()
    return None

def detect_text_file(path):
    """Detects text in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.text_detection(image=image)
    texts = response.text_annotations
    print('Texts:')

    for text in texts:
        print('\n"{}"'.format(text.description))

        vertices = (['({},{})'.format(vertex.x, vertex.y)
                    for vertex in text.bounding_poly.vertices])

        print('bounds: {}'.format(','.join(vertices)))

def detect_safe_search_numpy(image):
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation
    
    return safe.adult

def detect_safe_search_file(path):
    """Detects unsafe features in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation
    
    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    #print('Safe search:')
    #print('adult: {}'.format(likelihood_name[safe.adult]))
    #print('medical: {}'.format(likelihood_name[safe.medical]))
    #print('spoofed: {}'.format(likelihood_name[safe.spoof]))
    #print('violence: {}'.format(likelihood_name[safe.violence]))
    #print('racy: {}'.format(likelihood_name[safe.racy]))
    return (safe.adult, safe.racy)

def detect_faces_numpy(image):
    """Detects faces in an image."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    response = client.face_detection(image=image)
    faces = response.face_annotations

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    if len(faces) == 0:
        return False

    return True

def detect_faces_file(path):
    """Detects faces in an image."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.face_detection(image=image)
    faces = response.face_annotations

    out_dic = {}
    out_dic['classes'] = []
    out_dic['boxes'] = []
    out_dic['scores'] = []
    for face in faces:
        out_dic['classes'].append('face')
        top = face.bounding_poly.vertices[0].y
        left = face.bounding_poly.vertices[0].x
        bottom = face.bounding_poly.vertices[2].y
        right = face.bounding_poly.vertices[2].x
        bbox = [top, left, bottom, right]
        out_dic['boxes'].append(bbox)
        out_dic['scores'].append(face.detection_confidence)
    return out_dic

def googleDet_to_Dictionary(google_det, image_hw):
    '''transfer google object detection output to dictrionary of lists.
    Output:
        {
            'boxes' : [[top, left, bottom, right], ...]
            'scores' : [float, ...]
            'classes' : [int, ...]
        }

    '''
    H, W = image_hw
    dic_out = {}
    if google_det is None or len(google_det) == 0:
        return dic_out
    boxes_list = []
    scores_list = []
    classes_list = []
    for temp_det in google_det:
        temp_name = temp_det.name
        temp_score = temp_det.score
        left = temp_det.bounding_poly.normalized_vertices[0].x * W
        top = temp_det.bounding_poly.normalized_vertices[0].y * H
        right = temp_det.bounding_poly.normalized_vertices[2].x * W
        bottom = temp_det.bounding_poly.normalized_vertices[2].y * H
        boxes_list.append([top, left, bottom, right])
        scores_list.append(temp_score)
        classes_list.append(temp_name)
    dic_out['boxes'] = boxes_list
    dic_out['scores'] = scores_list
    dic_out['classes'] = classes_list
    return dic_out
