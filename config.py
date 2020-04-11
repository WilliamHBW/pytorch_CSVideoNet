# coding:utf8
import warnings
import torch as t

class DefaultConfig(object):
    '''
    env = 'default'  # visdom 环境
    vis_port =8097 # visdom 端口
    model = 'SqueezeNet'  # 使用的模型，名字必须与models/__init__.py中的名字一致

    train_data_root = './data/train/'  # 训练集存放路径
    test_data_root = './data/test1'  # 测试集存放路径
    load_model_path = None  # 加载预训练的模型的路径，为None代表不加载

    batch_size = 32  # batch size
    use_gpu = True  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    debug_file = '/tmp/debug'  # if os.path.exists(debug_file): enter ipdb
    result_file = 'result.csv'

    max_epoch = 10
    lr = 0.001  # initial learning rate
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0e-5  # 损失函数
    '''
    train_data_root = './data/train'
    test_data_root = './data/test'
    load_model_path = None
    save_test_root = './result'

    batch_size = 200  # batch size
    use_gpu = False  # user GPU or not
    num_workers = 4  # how many workers for loading data
    print_freq = 20  # print info every N batch

    max_epoch = 10000
    lr = 1e-4 # initial learning rate
    momentum = 0.9
    lr_decay = 0.5  # when val_loss increase, lr = lr*lr_decay
    lr_decay_ever = 3
    weight_decay = 5e-4  # 损失函数

    #model related parameters
    CR = [1,0.5]
    seqLength = 10
    Height = 32
    Width = 32
    overlap = 0 #帧分割块是否重叠

    gradient_clipping = 10 #梯度剪枝
    device='cuda'
    gpu_aval = '5,6,7'

    def _parse(self, kwargs):
        """
        根据字典kwargs 更新 config参数
        """
        for k, v in kwargs.items():
            if not hasattr(self, k):
                warnings.warn("Warning: opt has not attribut %s" % k)
            setattr(self, k, v)
        
        #opt.device = t.device('cuda') if (opt.use_gpu and t.cuda.is_available()) else t.device('cpu')
        


        print('user config:')
        for k, v in self.__class__.__dict__.items():
            if not k.startswith('_'):
                print(k, getattr(self, k))

opt = DefaultConfig()
