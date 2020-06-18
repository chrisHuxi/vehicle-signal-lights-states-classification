def load_model():
    model = None
    if config.snapshot is None:
        elif config.CNN_LSTM:
            print("loading CNN_LSTM model......")
            model = CNN_LSTM(config)
            shutil.copy("./models/model_CNN_LSTM.py", "./snapshot/" + config.mulu)

        print(model)
    else:
        print('\nLoading model from [%s]...' % config.snapshot)
        try:
            model = torch.load(config.snapshot)
        except:
            print("Sorry, This snapshot doesn't exist.")
            exit()
    if config.cuda is True:
        model = model.cuda()
    return model


def start_train(model, train_iter, dev_iter, test_iter):
    """
    :function：start train
    :param model:
    :param train_iter:
    :param dev_iter:
    :param test_iter:
    :return:
    """
    if config.predict is not None:
        label = train_ALL_CNN.predict(config.predict, model, config.text_field, config.label_field)
        print('\n[Text]  {}[Label] {}\n'.format(config.predict, label))
    elif config.test:
        try:
            print(test_iter)
            train_ALL_CNN.test_eval(test_iter, model, config)
        except Exception as e:
            print("\nSorry. The test dataset doesn't  exist.\n")
    else:
        print("\n cpu_count \n", mu.cpu_count())
        torch.set_num_threads(config.num_threads)
        if os.path.exists("./Test_Result.txt"):
            os.remove("./Test_Result.txt")

        elif config.CNN_LSTM:
            print("CNN_LSTM training start......")
            model_count = train_ALL_LSTM.train(train_iter, dev_iter, test_iter, model, config)

        print("Model_count", model_count)
        resultlist = []
        if os.path.exists("./Test_Result.txt"):
            file = open("./Test_Result.txt")
            for line in file.readlines():
                if line[:10] == "Evaluation":
                    resultlist.append(float(line[34:41]))
            result = sorted(resultlist)
            file.close()
            file = open("./Test_Result.txt", "a")
            file.write("\nThe Best Result is : " + str(result[len(result) - 1]))
            file.write("\n")
            file.close()
            shutil.copy("./Test_Result.txt", "./snapshot/" + config.mulu + "/Test_Result.txt")


def main():
    """
        main function
    """
    # define word dict
    define_dict()
    # load data
    train_iter, dev_iter, test_iter = Load_Data()
    # load pretrain embedding
    load_preEmbedding()
    # update config and print
    update_arguments()
    save_arguments()
    model = load_model()
    start_train(model, train_iter, dev_iter, test_iter)


if __name__ == "__main__":
    print("Process ID {}, Process Parent ID {}".format(os.getpid(), os.getppid()))
    parser = argparse.ArgumentParser(description="Neural Networks")
    parser.add_argument('--config_file', default="./Config/config.cfg")
    config = parser.parse_args()

    config = configurable.Configurable(config_file=config.config_file)
    if config.cuda is True:
        print("Using GPU To Train......")
        # torch.backends.cudnn.enabled = True
        torch.backends.cudnn.deterministic = True
        torch.cuda.manual_seed(seed_num)
        torch.cuda.manual_seed_all(seed_num)
        print("torch.cuda.initial_seed", torch.cuda.initial_seed())
    main()
