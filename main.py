import A.cnn_aa as ACNN
import A.svm_aa as ASVM
import B.svm_bb as BSVM
import B.cnn_bb as BCNN


def main():
    print("Choose the Task:")
    task=int(input("1. Task A\n2. Task B\n").strip())
    model = int(input("\nWhich model would you like to use\n1. SVM\n2. CNN\n"))

    if task == 1:
        if model == 1:
            ASVM.load_test()
        elif model == 2:
            ACNN.load_test()
        else:
            raise(ValueError)
    elif task == 2:
        if model == 1:
            BSVM.load_test()
        elif model == 2:
            BCNN.load_test()
        else:
            raise(ValueError)


if __name__ == "__main__":
    main()