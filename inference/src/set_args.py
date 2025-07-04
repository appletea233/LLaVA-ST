import yaml

config_path = "./inference/config.yaml"
with open(file=config_path, mode='r', encoding='utf-8') as fp:
    config = yaml.load(stream=fp, Loader=yaml.FullLoader)


def set_args(args):
    if args.all_name == "refcoco":
        args.ann_path = config["refcoco"]["basic"]
        args.data_folder = config["refcoco"]["data_folder"]
        args.task = "rec"
        args.dataset = "refcoco"
    elif args.all_name == "refcoco_testA":
        args.ann_path = config["refcoco"]["basic_A"]
        args.data_folder = config["refcoco"]["data_folder"]
        args.task = "rec"
        args.dataset = "refcoco"
    elif args.all_name == "refcoco_testB":
        args.ann_path = config["refcoco"]["basic_B"]
        args.data_folder = config["refcoco"]["data_folder"]
        args.task = "rec"
        args.dataset = "refcoco"
    elif args.all_name == "refcoco+":
        args.ann_path = config["refcoco"]["add"]
        args.data_folder = config["refcoco"]["data_folder"]
        args.task = "rec"
        args.dataset = "refcoco"
    elif args.all_name == "refcoco+_testA":
        args.ann_path = config["refcoco"]["add_A"]
        args.data_folder = config["refcoco"]["data_folder"]
        args.task = "rec"
        args.dataset = "refcoco"
    elif args.all_name == "refcoco+_testB":
        args.ann_path = config["refcoco"]["add_B"]
        args.data_folder = config["refcoco"]["data_folder"]
        args.task = "rec"
        args.dataset = "refcoco"
    elif args.all_name == "refcocog":
        args.ann_path = config["refcoco"]["g"]
        args.data_folder = config["refcoco"]["data_folder"]
        args.task = "rec"
        args.dataset = "refcoco"
    elif args.all_name == "refcocog_test":
        args.ann_path = config["refcoco"]["g_"]
        args.data_folder = config["refcoco"]["data_folder"]
        args.task = "rec"
        args.dataset = "refcoco"
    elif args.all_name == "charades_sta":
        args.ann_path = config["charades"]["sta"]
        args.data_folder = config["charades"]["data_folder"]
        args.task = "tvg"
        args.dataset = "charades_sta"
    elif args.all_name == "stvg":
        args.ann_path = config["st-align"]["stvg"]
        args.data_folder = config["st-align"]["data_folder"]
        args.task = "st-align"
        args.task_id = 0
        args.dataset = "vidstg"
    elif args.all_name == "svg":
        args.ann_path = config["st-align"]["svg"]
        args.data_folder = config["st-align"]["data_folder"]
        args.task = "st-align"
        args.task_id = 1
        args.dataset = "vidstg"
    elif args.all_name == "elc":
        args.ann_path = config["st-align"]["elc"]
        args.data_folder = config["st-align"]["data_folder"]
        args.task = "st-align"
        args.task_id = 2
        args.dataset = "vidstg"
