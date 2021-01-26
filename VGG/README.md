# Chansey
This is the vgg discriminator between sCT images and CT images. The size of input images is 512x512x3.

data folder info: this folder contains the data used for training and validation, saved model, and the test data
	 |	
	  --- per 
	  	   |	
  		    --- gp_1
				|	
	  		     --- 10_cases
						|	
	  		     		 --- data_for_dis (the data used to train VGG16)
						|	
	  		     		 --- data_for_gen (the data used to train Unet)
						|	
	  		     		 --- test (the data used for prediction)
						|	
	  		     		 --- 1st_round (the saved model in the first round training)
								|	
		  		     		 	 --- dis (saved VGG model)
								|	
		  		     		 	 --- gen (saved Unet model)
						|	
	  		     		 --- 2nd_round (the saved model in the second round training)
				|	
		  		 --- 20_cases
				|	
		  		 --- 30_cases
	  	   |	
  		    ---  gp_2
		   |	
  		    --- ...
	  	   |	
  		    ---  gp_10

	