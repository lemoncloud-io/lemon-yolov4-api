# coding=utf-8
"""
download images and make yolo lables.

- simple run
`$ python -m shoes.download --image 1`

- download jpg
`$ python -m shoes.download --url 'https://shopping-phinf.pstatic.net/main_1722383/17223830624.20190126230453.jpg'`


#TODO
- FileNotFoundError: [Errno 2] No such file or directory: '/training/predictions.jpg'

@copyright  lemoncloud.io 2020
"""
import os, json, time
from os.path import join
from urllib import request
import requests
from io import BytesIO
from PIL import Image
from absl import app, flags

flags.DEFINE_string('agent', 'bubble', 'Agent to visualize.')
flags.DEFINE_integer('image',   0, 'flag to download images')
flags.DEFINE_integer('yolo',    1, 'flag to make yolo txt file')
flags.DEFINE_string('url', '', 'image url to download.')
FLAGS = flags.FLAGS

# curren folder.
def curr_dir():
    dir_path = os.path.dirname(os.path.realpath(__file__))
    return dir_path

def loadconf(base: str = ''):
    base = os.path.join(curr_dir()) if not base else base
    with open(os.path.join(base, 'train_config.json')) as json_file:
        return json.load(json_file)

# load json
def loadjson(name: str, base: str = '', classes = None):
    base = os.path.join(curr_dir(), 'labels', 'json') if not base else base
    with open(os.path.join(base, name)) as json_file:
        data = json.load(json_file)
        id = data['id']
        img = data['context']['image']
        url = img['url']
        w = img['width']
        h = img['height']
        tag = [(A['label']['name'],A['rect']['x'],A['rect']['y'],A['rect']['width'],A['rect']['height']) for A in data['annotations']]
        new = load_def(id, base)
        tag = new if new else tag
        translate = lambda x: 'sliper' if x == '슬리퍼' else 'soccer' if x == '축구화' else 'running' if x == '운동화' else x
        # <object-class> <x_center> <y_center> <width> <height>
        if classes:
            tag = [(classes.index(translate(t[0])), (t[1]+t[3]/2)/w, (t[2]+t[4]/2)/h, t[3]/w, t[4]/h) for t in tag]
        return { 'id': id, 'url': url, 'w':w, 'h':h, 'tag':tag }

def load_def(id: str, base: str):
    with open(os.path.join(base, '..', 'images.json')) as f:
        data = json.load(f)
        found = [d for d in data if d['image'] == '{}.jpg'.format(id)]
        found = found[0] if len(found) > 0 else None
        if not found: return None
        key = 'coordinates'
        tag = [(A['label'], A[key]['x']-A[key]['width']/2, A[key]['y']-A[key]['height']/2, A[key]['width'], A[key]['height']) for A in found['annotations']]
        return tag

def tag2lines(tag):
    return [(lambda t: ' '.join(['%d'%t[0], '%.7f'%t[1], '%.7f'%t[2], '%.7f'%t[3], '%.7f'%t[4]]))(tag[i]) for i in range(len(tag))]

# save to yolo.txt file
def saveyolo(info: dict, base: str = ''):
    base = os.path.join(curr_dir(), 'labels', 'yolo') if not base else base
    if not os.path.exists(base): os.makedirs(base)
    id = info['id']
    tag = info['tag']
    name = '{}.{}'.format(id, 'txt')
    with open(os.path.join(base, name), 'w') as f:
        f.writelines('\n'.join(tag2lines(tag)))
        f.write('\n')
    return name

# download url
def download(url: str, id: str = ''):
    name = url.split('/')[-1]
    ext = name.split('.')[-1].lower()
    name = '{}.{}'.format(id, ext) if id else name
    base = os.path.join(curr_dir(), 'images')
    if not os.path.exists(base): os.makedirs(base)

    # time check
    start = time.time()
    opener = request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    request.install_opener(opener)    
    request.urlretrieve(url, os.path.join(base, name))
    # res1 = request.urlopen(url).read()
    # res2 = requests.get(url)
    elapsed = time.time() - start
    print('> down[%s] elapsed = %.2f'%(name, elapsed))

    # read image
    # img = Image.open("test.jpg")
    # img = Image.open(BytesIO(res1))
    # img = Image.open(BytesIO(res2.content))
    return name, base, ext

def down_url(url: str):
    name = url.split('/')[-1]
    id = name.split('.')[0]

    file, base, ext = download(url, id)
    file = os.path.join(base, file)
    img = Image.open(file)
    w,h = img.width, img.height

    data = {
        'id': id,
        'context':{
            'type': 'image',
            'contentType': 'image/{}'.format('jpeg' if ext == 'jpg' else ext),
            'image':{
                'url': url,
                'width': w,
                'height': h,
            }
        },
        'annotations':[],
    }
    conf = os.path.join(curr_dir(), 'labels', 'json')
    with open(os.path.join(conf, '{}.json'.format(id)), 'w') as f:
        json.dump(data, f, indent=4)
    print('> down-url[{}][{}x{}] ='.format(name,w,h), '{}.json'.format(id))
    return id

# run main
def main(_, hello='lemon-yolov4'):
    print('main({})'.format(hello))
    dir = curr_dir()
    print('> dir =', dir)
    # download('https://lemon-hello-www.s3.ap-northeast-2.amazonaws.com/image/2f20acbe-fc4f-405a-8ebd-2713721510b5.jpg')
    # download('https://shopping-phinf.pstatic.net/main_1782121/17821212487.20190303180108.jpg')
    if FLAGS.url:
        down_url(FLAGS.url)
        return

    # print('> json =', loadjson('0bc7b31a-6e87-40f1-b363-affe7265b01a.json'))
    conf = loadconf()
    classes = conf['data']['classes']
    base = os.path.join(curr_dir(), 'labels', 'json')
    jsonfiles = [f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))]
    print('> jsonfiles[0] =', jsonfiles[0])

    jsoninfos = [loadjson(f, base, classes) for f in jsonfiles]
    print('> jsoninfos[0] =', jsoninfos[0])

    #! write yolo files.
    if FLAGS.yolo:
        yolofiles = [saveyolo(f) for f in jsoninfos]
        print('> labels.len =', len(yolofiles))

    #! download images to `images` folder.
    if FLAGS.image:
        images = [download(f['url'], f['id']) for f in jsoninfos]
        print('> images.len =', len(images))
# main
if __name__ == '__main__':
    app.run(main)