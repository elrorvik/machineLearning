from OCR import *
from extract_characters import *

def classiify(model,pp,images):

    feature_list = []
    for i in range(len(images)):
        image = images[i]
        label = labels[i][0]

        df = hog_feature_extraction(image)
        df = pp.transform(np.array([df],'float64'))

        feature_list.append(df)

    results = []
    for i in range(len(feature_list)):
        #Prediction
        predict = model.predict(feature_list[i].reshape((1,-1)))[0]
        prob = model.predict_proba(feature_list[i].reshape((1,-1)))
        results.append([images[i],predict,prob.max()])

    return results


    


def main():
    #----------Training-------------
    col_dir = 'chars74k-lite/*/*.jpg'
    print("===Training: ",col_dir)
    label,image = get_image(col_dir)

    train_images,test_images,train_labels,test_labels = split_train_test(label,image)

    train_images = data_processing(train_images)
    test_images = data_processing(test_images)
    model,pp = training(train_images, train_labels)
    test(model,pp,test_images,test_labels)

    #---------Classify Images-----------
    clas_dir = 'detection-images/*.jpg'
    print("===Classify: ",clas_dir )
    clas_images = get_image_classify(clas_dir)
    class_segments = extract_window(clas_images[1],name="image2",save=True)
    results = classify(model,pp,class_segments)
    plt.figure()
    plt.imshow(results[0][0])
    print(results[0][1],results[0][2])
    plt.show()
    

    #plt.figure()
    #plt.imshow(train_images[4000])
    #plt.figure()
    #plt.imshow(train_images[10])
    #print(label_to_letter(train_labels[4000]))
    #print(label_to_letter(train_labels[10]))
    #plt.show()


if __name__ == "__main__":
    main()

