import os
import glob
import pandas as pd
import io
import xml.etree.ElementTree as ET
import argparse

import tensorflow as tf
from PIL import Image
from object_detection.utils import dataset_util, label_map_util
from collections import namedtuple

parser = argparse.ArgumentParser(description="Generate TFRecord from XML and images")
parser.add_argument("-x", "--xml_dir", type=str, help="Path to XML directory")
parser.add_argument("-l", "--labels_path", type=str, help="Path to label map (.pbtxt)")
parser.add_argument("-o", "--output_path", type=str, help="Path to output TFRecord")
parser.add_argument("-i", "--image_dir", type=str, default=None, help="Path to image directory")
parser.add_argument("-c", "--csv_path", type=str, default=None, help="Path to optional CSV output")
args = parser.parse_args()

if args.image_dir is None:
    args.image_dir = args.xml_dir

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(os.path.join(path, '*.xml')):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (
                root.find('filename').text,
                int(root.find('size/width').text),
                int(root.find('size/height').text),
                member.find('name').text,
                int(member.find('bndbox/xmin').text),
                int(member.find('bndbox/ymin').text),
                int(member.find('bndbox/xmax').text),
                int(member.find('bndbox/ymax').text)
            )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    return pd.DataFrame(xml_list, columns=column_name)

def class_text_to_int(row_label, label_map_dict):
    return label_map_dict.get(row_label, None)

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def create_tf_example(group, path, label_map_dict):
    with tf.io.gfile.GFile(os.path.join(path, group.filename), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins, xmaxs, ymins, ymaxs, classes_text, classes = [], [], [], [], [], []

    for _, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class'], label_map_dict))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

def main():
    # Use Python's built-in file reading for label map
    with open(args.labels_path, 'r') as file:
        label_map_data = file.read()
    
    label_map_dict = label_map_util.get_label_map_dict(args.labels_path)

    writer = tf.io.TFRecordWriter(args.output_path)

    examples = xml_to_csv(args.xml_dir)
    grouped = split(examples, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, args.image_dir, label_map_dict)
        writer.write(tf_example.SerializeToString())
    writer.close()

    print(f'Successfully created the TFRecord file: {args.output_path}')
    if args.csv_path:
        examples.to_csv(args.csv_path, index=False)
        print(f'Successfully created the CSV file: {args.csv_path}')

if __name__ == '__main__':
    main()
