# coding=utf-8
"""
download images and make yolo lables.

- simple run
`$ python -m shoes.download --image 1`

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
flags.DEFINE_integer('image', 0, 'flag to download images')
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
        if classes:
            tag = [(classes.index(t[0]), t[1]/w, t[2]/h, t[3]/w, t[4]/h) for t in tag]
        return { 'id': id, 'url': url, 'w':w, 'h':h, 'tag':tag }

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
    ext = name.split('.')[-1]
    name = '{}.{}'.format(id, ext) if id else name
    base = os.path.join(curr_dir(), 'images')
    if not os.path.exists(base): os.makedirs(base)

    # time check
    start = time.time()
    request.urlretrieve(url, os.path.join(base, name))
    # res1 = request.urlopen(url).read()
    # res2 = requests.get(url)
    elapsed = time.time() - start
    print('> down[%s] elapsed = %.2f'%(name, elapsed))

    # read image
    # img = Image.open("test.jpg")
    # img = Image.open(BytesIO(res1))
    # img = Image.open(BytesIO(res2.content))
    return elapsed, name

# run main
def main(_, hello='Hello'):
    print('main()')
    dir = curr_dir()
    print('> dir =', dir)
    # download('https://lemon-hello-www.s3.ap-northeast-2.amazonaws.com/image/2f20acbe-fc4f-405a-8ebd-2713721510b5.jpg')
    # print('> json =', loadjson('0bc7b31a-6e87-40f1-b363-affe7265b01a.json'))
    # print('> json =', loadjson('3f1634a3-f76c-4969-9728-fcb1009e40e5.json'))
    conf = loadconf()
    # print('> conf =', conf)
    classes = conf['data']['classes']
    base = os.path.join(curr_dir(), 'labels', 'json')
    jsonfiles = [f for f in os.listdir(base) if os.path.isfile(os.path.join(base, f))]
    # print('> jsonfiles =', jsonfiles)
    # print('> loadjson =', loadjson('da3ad406-a517-4518-b1f2-ba6a51471379.json', base, classes))
    # print('> lines = ', tag2lines(loadjson('da3ad406-a517-4518-b1f2-ba6a51471379.json', base, classes)['tag']))

    jsoninfos = [loadjson(f, base, classes) for f in jsonfiles]
    print('> jsoninfos[0] =', jsoninfos[0])
    # jsonyolos = [saveyolo(f) for f in jsoninfos]
    # print('> jsonyolos[0] =', jsonyolos[0])
    # jsoninfos = [loadjson(f, base, classes) for f in jsonfiles]
    # print('> jsonyolos[0] =', saveyolo(loadjson('da3ad406-a517-4518-b1f2-ba6a51471379.json', base, classes)))
    #! write yolo files.
    if True:
        yolofiles = [saveyolo(f) for f in jsoninfos]
        print('> labels.len =', len(yolofiles))

    #! download images to `images` folder.
    if FLAGS.image:
        images = [download(f['url'], f['id']) for f in jsoninfos]
        print('> images.len =', len(images))
# main
if __name__ == '__main__':
    app.run(main)