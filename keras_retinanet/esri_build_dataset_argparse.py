# Construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-a", "--annotations", default=config.ANNOT_PATH,
    help='path to annotations')
ap.add_argument("-i", "--images", default=config.IMAGES_PATH,
	help="path to images")
ap.add_argument("-t", "--train", default=config.TRAIN_CSV,
	help="path to output training CSV file")
ap.add_argument("-e", "--test", default=config.TEST_CSV,
	help="path to output test CSV file")
ap.add_argument("-c", "--classes", default=config.CLASSES_CSV,
	help="path to output classes CSV file")
ap.add_argument("-s", "--split", type=float, default=config.TRAIN_TEST_SPLIT,
	help="train and test split")
args = vars(ap.parse_args())

# Create easy variable names for all the arguments
annot_path = args["annotations"]
images_path = args["images"]
train_csv = args["train"]
test_csv = args["test"]
classes_csv = args["classes"]
train_test_split = args["split"]