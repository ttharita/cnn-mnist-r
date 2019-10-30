install.packages("keras")
library(keras)
library(tensorflow)
#install_tensorflow() 

batch_size <- 128 
num_classes <- 10 # digits from 0 to 9 
epochs <- 1 

# dimension 
img_rows <- 28
img_cols <- 28

# load mnist dataset

mnist <- dataset_mnist() 
x_train <- mnist$train$x 
y_train <- mnist$train$y 
x_test <- mnist$test$x
y_test <- mnist$test$y

# reshape the dimension of train/test inputs

#array_reshape(x, dim) with x as an array to be reshaped with the new dim dimension
x_train <- array_reshape(x_train, c(nrow(x_train), img_rows, img_cols, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), img_rows, img_cols, 1))
input_shape <- c(img_rows, img_cols, 1)

# normalize (x_train and x_test will be in the range between 0 to 1)
x_train <- x_train / 255
x_test <- x_test / 255

cat('x_train_shape:', dim(x_train), '\n')
cat(nrow(x_train), 'train samples\n')
cat(nrow(x_test), 'test samples\n')


y_train <- to_categorical(y_train, num_classes) 
# with dimension 60,000 x 10 , each column for each class and the correct label (class)
# to each observation (each row) will be assigned 1 in that particular class (that column)

y_test <- to_categorical(y_test, num_classes)
# analogously with y_test 


# just checking out the dimension 
dim(y_train)
dim(y_test)

# our CNN model 
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu',
                input_shape = input_shape) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_dropout(rate = 0.25) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = num_classes, activation = 'softmax')

# compile the model
model %>% compile(
  loss = loss_categorical_crossentropy,
  optimizer = optimizer_adadelta(),
  metrics = c('accuracy')
)

# train model
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  validation_split = 0.2 
)


scores <- model %>% evaluate(
  x_test, y_test, verbose = 0
)

# metrics 
cat('Test input loss:', scores[[1]], '\n')
cat('Test input accuracy:', scores[[2]], '\n')


