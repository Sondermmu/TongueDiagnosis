import os
import torch

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 数据目录
DATA_DIR = os.path.join(ROOT_DIR, 'data')
SEG_DATA_DIR = os.path.join(DATA_DIR, 'seg_data')
CLS_DATA_DIR = os.path.join(DATA_DIR, 'cls_data')
CROP_DATA_DIR = os.path.join(DATA_DIR, 'seg_crop_data')

# 输出目录
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
SEG_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'segmentation')
CLS_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'classification')

# 临时目录
TEMP_DIR = os.path.join(ROOT_DIR, 'temp')

# 分割模型配置
SEG_CONFIG = {
    "BATCH_SIZE": 32,
    "EPOCHS": 100,
    "IMAGE_SIZE": 224,
    "CLASS": 2,
    "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "ENCODER": "resnet34",
    "ENCODER_WEIGHTS": "imagenet",
    "LEARNING_RATE": 1e-4,
    "PATIENCE": 10,
    "SEED": 66,
    "MODEL_PATH": os.path.join(SEG_OUTPUT_DIR, 'weights', 'seg.pth'),
    "BEST_MODEL_PATH": os.path.join(SEG_OUTPUT_DIR, 'weights', 'best_model.pth'),
    "LOG_FILE": os.path.join(SEG_OUTPUT_DIR, 'logs', 'train_log.log'),
    "PLOTS_DIR": os.path.join(SEG_OUTPUT_DIR, 'plots')
}

# 分类模型配置
CLS_CONFIG = {
    "BATCH_SIZE": 32,
    "EPOCHS": 100,
    "IMAGE_SIZE": 224,
    "DEVICE": torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    "LEARNING_RATE": 1e-4,
    "PATIENCE": 10,
    "NUM_WORKERS": 4,
    "PLOTS_DIR": os.path.join(CLS_OUTPUT_DIR, 'plots'),
    "MODEL_PATH": os.path.join(CLS_OUTPUT_DIR, 'weights'),
    "LOG_FILE": os.path.join(CLS_OUTPUT_DIR, 'logs', 'train_log.log'),
    "SEED": 42
}

# 类别名称
CLASS_NAMES = ['舌红、苔厚腻', '舌白、苔厚腻', '黑苔', '地图舌', '紫苔', '舌红、黄苔、厚腻苔']

# 舌象建议
TONGUE_ADVICE = {
    '黑苔': {
        '舌象': '多为重病、久病表现，主里证 ，常见寒盛或热极。\n黑苔干燥是热极津伤；黑苔滑润是阳虚寒盛。也可能因精神高度紧张、经常熬夜、抽烟过多等导致 ，还可能由肠胃毛病生寒湿，舌苔从白逐渐变黄或变黑。\n部分慢性病恶化时也会出现，如尿毒症、恶性肿瘤等。',
        '建议': '1. 及时就医检查\n2. 注意休息，避免过度劳累\n3. 注意舌苔颜色可能受饮食、吸烟、药物等影响，观察时综合考虑其他症状体征及生活习惯 '
    },
    '地图舌': {
        '舌象': '舌面出现不规则的红斑，边缘微微隆起，形状类似地图。\n它与消化不良、营养缺乏（如缺乏锌、铁、维生素 B 族 ）、肠道寄生虫、过敏体质、精神心理因素（如情绪紧张、焦虑 ）等有关，也可能是某些全身性疾病（如银屑病、干燥综合征 ）在口腔的表现。',
        '建议': '1. 保持口腔卫生，早晚刷牙、饭后漱口\n2. 均衡饮食，多吃富含维生素和矿物质的食物，避免挑食\n3. 规律作息，保证充足睡眠，避免过度劳累\n4. 若由全身性疾病引起，积极治疗原发病；若因精神因素，可通过运动、听音乐等方式缓解压力。'
    },
    '紫苔': {
        '舌象': '舌色呈现紫色。\n全舌紫多为气血运行不畅，可能是寒凝血瘀、热盛血瘀等。舌有紫色斑点，可能是瘀血阻滞局部。常见于心血管疾病（如冠心病、心肌梗死 ）、呼吸系统疾病（如慢性阻塞性肺疾病 ）、血液系统疾病（如血小板减少性紫癜 ）等患者。',
        '建议': '1. 及时就医，进行全面检查，明确病因。\n2. 若是寒凝血瘀，可适当食用温热性食物，如羊肉、桂圆等，注意保暖\n3. 若是热盛血瘀，饮食清淡，多吃清热凉血食物，如莲藕、生地黄等\n4. 若由疾病引起，遵医嘱治疗相关疾病'
    },
    '舌红、黄苔、厚腻苔': {
        '舌象': '舌红主热证，黄苔主里证、热证 ，厚腻苔多与湿热、痰热内蕴或食积化腐有关。\n可能因平时饮食过量、进食过于油腻食物，导致体内湿热或痰热积聚，或食积不化、食物腐败生热。',
        '建议': '1. 饮食清淡，多吃蔬菜水果，如冬瓜、苦瓜、芹菜等清热利湿食物 ，避免辛辣、油腻、刺激性食物\n2. 可适当服用清热祛湿药物，但需在医生指导下进行\n3. 适当运动，促进代谢\n4. 规律作息，避免熬夜'
    },
    '舌红、苔厚腻': {
        '舌象': '舌红提示热证，厚腻苔提示体内有湿浊、痰饮或食积。\n可能是体内湿热蕴结，也可能是脾胃功能失调，运化水湿和食物功能减弱，导致水湿和食物积滞化热。',
        '建议': '1. 调整饮食，少吃肥甘厚味、生冷食物 ，多吃健脾祛湿食物，如薏苡仁、芡实、白扁豆等\n2. 规律生活作息，避免暴饮暴食和过度劳累\n3. 可尝试中医理疗，如艾灸足三里、中脘等穴位，起到健脾祛湿作用\n4. 若症状持续或加重，及时就医'
    },
    '舌白、苔厚腻': {
        '舌象': '白苔一般提示表证、寒证 ，厚腻苔表示湿浊内停、痰饮或食积。\n常见于常吃生冷食物、喝冷饮人群，或体型肥胖、水湿代谢不畅者，也可能是脾胃虚弱，不能正常运化水湿和食物。',
        '建议': '1. 注意保暖，尤其是腹部保暖，避免受寒\n2. 饮食选择温热、易消化食物，如小米粥、山药粥等 ，少吃生冷、油腻食物\n3. 可在医生指导下服用健脾祛湿中药调理；适当运动，增强体质，促进水湿代谢'
    }
}


# 创建必要的目录
def create_directories():
    """创建项目所需的所有目录"""
    directories = [
        # 输出目录
        os.path.join(SEG_OUTPUT_DIR, 'weights'),
        os.path.join(SEG_OUTPUT_DIR, 'logs'),
        os.path.join(SEG_OUTPUT_DIR, 'plots'),
        os.path.join(CLS_OUTPUT_DIR, 'weights'),
        os.path.join(CLS_OUTPUT_DIR, 'logs'),
        os.path.join(CLS_OUTPUT_DIR, 'plots'),

        # 临时目录
        TEMP_DIR
    ]

    for directory in directories:
        os.makedirs(directory, exist_ok=True)


# 初始化时创建目录
create_directories()