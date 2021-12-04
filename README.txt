
I started this project so I could have a classifier that would tell me whether or not an image has a person in it.
I'm happy with the 95% test set accuracy I was able to achieve, but I've included my training infrastructure incase you want to try and do better.

Here's a summary of everything:


folders:
	train_set <-- put images in here if you want to custom train the network
	test_set <-- put images in here to test during training 
	training_run_data <-- loss plots and .pkl files will automatically be saved here. 
			      New loss-plots won't rewrite old ones. New .pkl files will rewrite old ones unless the name is manually changed.
	photos_to_classify <-- put photos in here if you want my already-trained network to classify them
        people_folder <-- if network sees a person in the photo, they will copy the photo here
	no_people_folder <-- if network doesn't see a person, it will copy the photo here

files:
	train_classifier.ipynb <-- I wrote this to train the network. I use squeezenet v1.1 as a feature extractor, 
				   then feed those features into an MLP for better classification. This achieves about 95% test accuracy.
				   I tried many other networks as backbones, but squeezenet works best. As a bonus, it's also very light!

	use_classifier.ipynb <-- This is a fun little notebook if all you want to do is use the classifier.
				 To use it, first copy all your photos into the "photos_to_classify" folder. 
				 Then run the notebook and watch the classification happen! Photos with people will be put into the "people_folder" folder,
			         while photos without people will be put into the "no_people_folder" folder.
	my_utils.py <-- just a bunch of useful utilities that I've defined.




NOTE: if you don't have a gpu, change all occurences of 'cuda' to 'cpu', or else this won't work. (Or just run online thru Google Colab)