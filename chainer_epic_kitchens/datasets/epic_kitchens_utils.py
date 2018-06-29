import os

from chainer.dataset import download

root = 'pfnet/chainercv/epic_kitchens'


def get_epic_kitchens(year):
    data_root = download.get_dataset_directory(root)
    base_path = os.path.join(data_root, year)
    return base_path


epic_kitchens_bbox_label_names = (
    'pan',
    'pan:dust',
    'tap',
    'plate',
    'knife',
    'bowl',
    'spoon',
    'cupboard',
    'drawer',
    'fridge',
    'lid',
    'hand',
    'onion',
    'onion:spring',
    'pot',
    'glass',
    'water',
    'fork',
    'board:chopping',
    'bag',
    'sponge',
    'spatula',
    'cup',
    'oil',
    'bin',
    'meat',
    'potato',
    'bottle',
    'container',
    'tomato',
    'salt',
    'cloth',
    'sink',
    'door:kitchen',
    'pasta',
    'dish:soap',
    'food',
    'kettle',
    'box',
    'carrot',
    'sauce',
    'colander',
    'milk',
    'rice',
    'garlic',
    'pepper',
    'hob',
    'dough',
    'dishwasher',
    'egg',
    'cheese',
    'bread',
    'table',
    'salad',
    'microwave',
    'oven',
    'cooker:slow',
    'coffee',
    'filter',
    'jar',
    'rack:drying',
    'chicken',
    'tray',
    'mixture',
    'towel',
    'towel:kitchen',
    'peach',
    'skin',
    'courgette',
    'liquid:washing',
    'liquid',
    'leaf',
    'lettuce',
    'leaf:mint',
    'cutlery',
    'scissors',
    'package',
    'top',
    'spice',
    'tortilla',
    'paper',
    'machine:washing',
    'olive',
    'sausage',
    'glove:oven',
    'peeler:potato',
    'can',
    'mat',
    'mat:sushi',
    'vegetable',
    'wrap:plastic',
    'wrap',
    'flour',
    'cucumber',
    'curry',
    'cereal',
    'napkin',
    'soap',
    'squash',
    'fish',
    'chilli',
    'cover',
    'sugar',
    'aubergine',
    'jug',
    'heat',
    'leek',
    'rubbish',
    'ladle',
    'mushroom',
    'stock',
    'freezer',
    'light',
    'pizza',
    'ball',
    'yoghurt',
    'chopstick',
    'grape',
    'ginger',
    'banana',
    'oregano',
    'tuna',
    'kitchen',
    'salmon',
    'basket',
    'maker:coffee',
    'roll',
    'brush',
    'lemon',
    'clothes',
    'grater',
    'strainer',
    'bacon',
    'avocado',
    'blueberry',
    'pesto',
    'utensil',
    'bean:green',
    'floor',
    'lime',
    'foil',
    'grill',
    'ingredient',
    'scale',
    'paste:garlic',
    'processor:food',
    'nut:pine',
    'butter',
    'butter:peanut',
    'shelf',
    'timer',
    'rinse',
    'tablecloth',
    'switch',
    'powder:coconut',
    'powder:washing',
    'capsule',
    'oat',
    'tofu',
    'lighter',
    'corn',
    'vinegar',
    'grinder',
    'cap',
    'support',
    'cream',
    'content',
    'tongs',
    'pie',
    'fan:extractor',
    'raisin',
    'toaster',
    'broccoli',
    'pin:rolling',
    'plug',
    'button',
    'tea',
    'parsley',
    'flame',
    'herb',
    'base',
    'holder:filter',
    'thyme',
    'honey',
    'celery',
    'kiwi',
    'tissue',
    'time',
    'clip',
    'noodle',
    'yeast',
    'hummus',
    'coconut',
    'cabbage',
    'spinach',
    'nutella',
    'fruit',
    'dressing:salad',
    'omelette',
    'kale',
    'paella',
    'chip',
    'opener:bottle',
    'shirt',
    'chair',
    'sandwich',
    'burger:tuna',
    'pancake',
    'leftover',
    'risotto',
    'pestle',
    'sock',
    'pea',
    'apron',
    'juice',
    'wine',
    'dust',
    'desk',
    'mesh',
    'oatmeal',
    'artichoke',
    'remover:spot',
    'coriander',
    'mocha',
    'quorn',
    'soup',
    'turmeric',
    'knob',
    'seed',
    'boxer',
    'paprika',
    'juicer:lime',
    'guard:hand',
    'apple',
    'tahini',
    'finger',
    'salami',
    'mayonnaise',
    'biscuit',
    'pear',
    'mortar',
    'berry',
    'beef',
    'squeezer:lime',
    'tail',
    'stick:crab',
    'supplement',
    'phone',
    'shell:egg',
    'pith',
    'ring:onion',
    'cherry',
    'cake',
    'sprout',
    'almond',
    'mint',
    'flake:chilli',
    'cutter:pizza',
    'nesquik',
    'blender',
    'scrap',
    'backpack',
    'melon',
    'breadcrumb',
    'sticker',
    'shrimp',
    'smoothie',
    'grass:lemon',
    'ketchup',
    'slicer',
    'stand',
    'dumpling',
    'watch',
    'beer',
    'power',
    'heater',
    'basil',
    'cinnamon',
    'crisp',
    'asparagus',
    'drink',
    'fishcakes',
    'mustard',
    'caper',
    'whetstone',
    'candle',
    'control:remote',
    'instruction',
    'cork',
    'tab',
    'masher',
    'part',
    'muffin',
    'shaker:pepper',
    'garni:bouquet',
    'popcorn',
    'envelope',
    'chocolate',
    'spot',
    'window',
    'syrup',
    'bar:cereal',
    'croissant',
    'coke',
    'stereo',
    'alarm',
    'recipe',
    'handle',
    'sleeve',
    'cumin',
    'wire',
    'label',
    'fire',
    'presser',
    'air',
    'mouse',
    'boiler',
    'rest',
    'tablet',
    'poster',
    'trousers',
    'form',
    'rubber',
    'rug',
    'sheets',
    'pepper:cayenne',
    'waffle',
    'pineapple',
    'turkey',
    'alcohol',
    'rosemary',
    'lead',
    'book',
    'rim',
    'gravy',
    'straw',
    'hat',
    'cd',
    'slipper',
    'casserole',
    'ladder',
    'jambalaya',
    'wall',
    'tube',
    'lamp',
    'tarragon',
    'heart',
    'funnel',
    'whisk',
    'driver:s',
)
