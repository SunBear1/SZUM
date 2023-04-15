from matplotlib import pyplot as plt


def present_sample_images(images: list):
    sample_images = images[0:5] + images[1000:1005] + images[2000:2005] + images[3000:3005] + images[4000:4005]

    plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(sample_images[i][0], cmap=plt.cm.binary)
        plt.xlabel(sample_images[i][1])
    plt.title("Dataset sample")
    plt.show()

    fries_samples = images[5:15]

    plt.figure(figsize=(10, 10))
    plt.title("Easy patterns")
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(fries_samples[i][0], cmap=plt.cm.binary)
        plt.xlabel(fries_samples[i][1])
    plt.show()

    lasagna_samples = images[2050:2060]

    plt.figure(figsize=(10, 10))
    plt.title("Medium hard patterns")
    for i in range(10):
        plt.subplot(5, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(lasagna_samples[i][0], cmap=plt.cm.binary)
        plt.xlabel(lasagna_samples[i][1])
    plt.show()


def present_augmented_data(X, X_augmented):
    img1_orig = X[1000]
    img2_orig = X[3000]

    # Get the corresponding augmented images
    img1_aug = X_augmented[0][1000]
    img2_aug = X_augmented[0][3000]

    # Display the images
    fig, axs = plt.subplots(2, 2, figsize=(8, 8))

    axs[0, 0].imshow(img1_orig)
    axs[0, 0].set_title("Ice cream original")

    axs[0, 1].imshow(img1_aug)
    axs[0, 1].set_title("Ice cream augmented")

    axs[1, 0].imshow(img2_orig)
    axs[1, 0].set_title("Other original")

    axs[1, 1].imshow(img2_aug)
    axs[1, 1].set_title("Other augmented")

    plt.show()


def present_first_dataset_split(label_dict, y_train, y_val, y_test):
    # Calculate class counts for each dataset split
    train_counts = [len(y_train[y_train == label]) for label in label_dict.keys()]
    val_counts = [len(y_val[y_val == label]) for label in label_dict.keys()]
    test_counts = [len(y_test[y_test == label]) for label in label_dict.keys()]

    # Plot the class distribution for each dataset split
    fig, ax = plt.subplots()
    ax.bar(label_dict.keys(), train_counts)
    ax.set_title("Class Distribution in Training Set")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    plt.show()

    fig, ax = plt.subplots()
    ax.bar(label_dict.keys(), val_counts)
    ax.set_title("Class Distribution in Validation Set")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    plt.show()

    fig, ax = plt.subplots()
    ax.bar(label_dict.keys(), test_counts)
    ax.set_title("Class Distribution in Test Set")
    ax.set_xlabel("Class")
    ax.set_ylabel("Number of Images")
    plt.show()
