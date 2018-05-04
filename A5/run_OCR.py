from OCR import *
from extract_characters import *
from pathlib import Path
import pickle

def classify(model,pp,images):

    feature_list = []
    for i in range(len(images)):
        image = images[i]

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
    load = True
    #----------Training-------------
    col_dir = 'chars74k-lite/*/*.jpg'
    print("===Training: ",col_dir)
    label,image = get_image(col_dir)

    print("Processing training data")
    train_images,test_images,train_labels,test_labels = split_train_test(label,image)
    train_images = data_processing(train_images)
    test_images = data_processing(test_images)

    if(load and Path("saved_model.o").is_file() and Path("saved_pp.o").is_file()):
        print("Loading models")
        model = pickle.load(open("saved_model.o","rb"))
        pp = pickle.load(open("saved_pp.o","rb"))
    else:
        model,pp = training(train_images, train_labels,"HOG","SVM")
        pickle.dump(model, open("saved_model.o","wb"))
        pickle.dump(pp, open("saved_pp.o","wb"))

    print("Testing models")
    test(model,pp,test_images,test_labels)

    #---------Classify Images-----------
    clas_dir = 'detection-images/*.jpg'
    print("===Classify: ",clas_dir )
    clas_images = get_image_classify(clas_dir)
    class_segments = np.asarray(extract_window(clas_images[1],name="image2",save=True))
    class_preprocessed = data_processing(class_segments)
    results = classify(model,pp,class_preprocessed)
    t = 175
    print(results[t][1],results[t][2])
    plt.figure()
    plt.imshow(results[t][0])

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
