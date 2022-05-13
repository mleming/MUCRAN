from tensorflow.keras.layers import *
from tensorflow.keras.regularizers import *
from tensorflow.keras.models import Model

def get_encoder(imsize,args,n_intermed_features = 1024, latent_dim=512,ft_bank_baseline = 128):
	input_img = Input(shape = (imsize[0],imsize[1],imsize[2],1))
		
	feature = Conv3D(ft_bank_baseline, kernel_size=3,
	padding='same',strides=2)(input_img)
	feature = LeakyReLU(alpha=0.3)(feature)
	feature = BatchNormalization()(feature)
	
	feature = Conv3D(ft_bank_baseline*2, kernel_size=3,
		padding='same',strides=2)(feature)
	feature = LeakyReLU(alpha=0.3)(feature)
	feature = BatchNormalization()(feature)
	
	feature = Conv3D(ft_bank_baseline*2, kernel_size=3,
		padding='same',strides=2)(feature)
	feature = LeakyReLU(alpha=0.3)(feature)
	feature = BatchNormalization()(feature)
		
	feature = Conv3D(ft_bank_baseline*2, kernel_size=3,
		padding='same',strides=2)(feature)
	feature = LeakyReLU(alpha=0.3)(feature)
	feature = BatchNormalization()(feature)
		
	feature = Flatten()(feature)
	feature = Dense(latent_dim*4)(feature)
	feature = LeakyReLU(alpha=0.3)(feature)
	feature = BatchNormalization()(feature)
	
	feature = Dense(latent_dim*4)(feature)
	feature = LeakyReLU(alpha=0.3)(feature)
	feature = BatchNormalization()(feature)
	
	feature = Dense(n_intermed_features,
		kernel_regularizer=l2(1e-4))(feature)
	
	feature_dense = Flatten()(feature)
	encoder = Model(inputs=input_img,outputs=feature_dense)
	return encoder, input_img, feature_dense

def get_regressor(outsize,feature_dense,args,n_intermed_features=1024, latent_dim=512):

	inputs_x = Input(shape=(n_intermed_features,))
	feature = Dense(latent_dim*4,kernel_regularizer=l2(1e-4))(inputs_x)
	feature = LeakyReLU(alpha=0.3)(feature)
	if args.nobatch:
		feature = Dropout(0.3)(feature)
	else:
		feature = BatchNormalization()(feature)
	feature = Dense(latent_dim*2)(feature)
	feature = LeakyReLU(alpha=0.3)(feature)
	if args.nobatch:
		feature = Dropout(0.3)(feature)
	else:
		feature = BatchNormalization()(feature)
	feature = Dense(latent_dim*2,kernel_regularizer=l2(1e-4))(feature)
	feature = LeakyReLU(alpha=0.3)(feature)
	cf = Concatenate(axis=1)([Reshape((1, outsize[2]))\
		(Dense(outsize[2],activation='softmax',kernel_regularizer=l2(1e-4))(feature))\
		 for _ in range(outsize[1])])
	regressor = Model(inputs = inputs_x,outputs=cf)
	return regressor

