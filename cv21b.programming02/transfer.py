import json
from PIL import Image

with open('categories.txt') as f:
    categories_str = f.read()


categories = eval(categories_str)
train_dataset_path = './val'

f = open("val.json")
images = json.load(f)
img_index = 0
with open('./data/my_data/val.txt', 'w') as c:
    for img in images:
        all_info = images['' + img]
        objects = all_info['objects']
        single_img_record = str(img_index) + ' ' + train_dataset_path + '/' + img
        im = Image.open(train_dataset_path + '/' + img)
        data_error = False
        x = im.size[0]
        y = im.size[1]
        single_img_record = single_img_record + ' ' + str(im.size[0]) + ' ' + str(im.size[1])
        for obj in objects:
            obj_info = objects['' + obj]
            category = obj_info['category']
            category_index = categories.index(category)
            bbox = obj_info['bbox']
            if bbox[0] > x or bbox[2] > x or bbox[1] > y or bbox[3] > y:
                print(obj_info, x, y)
                data_error = True
            single_img_record = single_img_record + ' ' + str(category_index) + ' ' + str(bbox[0]) + ' ' + str(bbox[1]) + ' ' + str(bbox[2]) + ' ' + str(bbox[3])
        if data_error:
            continue
        single_img_record = single_img_record + '\n'
        img_index = img_index + 1
        print(single_img_record)
        c.write(single_img_record)


