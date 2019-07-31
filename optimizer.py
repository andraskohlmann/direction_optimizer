import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

# Load a toy dataset for the sake of this example
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess the data (these are Numpy arrays)
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
# Reserve 1000 samples for the optimization step finding
x_opt = x_train[-11000:-10000]
y_opt = y_train[-11000:-10000]

x_train = x_train[:-11000]
y_train = y_train[:-11000]

inputs = keras.Input(shape=(784,), name='digits')
x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
x = layers.Dense(64, activation='relu', name='dense_2')(x)
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
model = keras.Model(inputs=inputs, outputs=outputs)

# Instantiate an optimizer to train the model.
optimizer = keras.optimizers.SGD(lr=1e-3)
# Instantiate a loss function.
loss_fn = keras.losses.SparseCategoricalCrossentropy()

# Prepare the metrics.
train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
val_acc_metric = keras.metrics.SparseCategoricalAccuracy()
opt_acc_metric = keras.metrics.SparseCategoricalAccuracy()

# Prepare the training dataset.
batch_size = 64
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

# Prepare the validation dataset.
val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_dataset = val_dataset.batch(64)

# Prepare the optimization dataset.
opt_dataset = tf.data.Dataset.from_tensor_slices((x_opt, y_opt))
opt_dataset = opt_dataset.batch(64)

opt_steps = [1, 2]
# opt_steps = [1, 2, 4, 8, 18, 32]
# opt_steps = [1, 2, 4, 8, 18, 32, 64, 128]
# opt_steps = [1, 1e2, 1e3, 1e4, 1e5, 1e6]

summary_writer = tf.summary.create_file_writer('out/optimizer')
with summary_writer.as_default():
    # Iterate over epochs.
    for epoch in range(3):
        print('Start of epoch %d' % (epoch,))

        # Iterate over the batches of the dataset.
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                logits = model(x_batch_train)
                loss_value = loss_fn(y_batch_train, logits)
            grads = tape.gradient(loss_value, model.trainable_weights)

            min_loss = tf.float32.max
            for opt_step in opt_steps:
                grads = [g * opt_step for g in grads]
                opt_loss_value = 0
                optimizer.apply_gradients(zip(grads, model.trainable_weights))
                for x_batch_opt, y_batch_opt in opt_dataset:
                    opt_logits = model(x_batch_opt)
                    opt_loss_value += loss_fn(y_batch_opt, opt_logits)
                # print(opt_loss_value)
                if opt_loss_value < min_loss:
                    best_step_weights = model.get_weights()
                    min_loss = opt_loss_value

            model.set_weights(best_step_weights)
            # Update training metric.
            train_acc_metric(y_batch_train, logits)

            # Log every 200 batches.
            if step % 200 == 0:
                print('Training loss (for one batch) at step %s: %s' % (step, float(loss_value)))
                print('Seen so far: %s samples' % ((step + 1) * 64))

        # Display metrics at the end of each epoch.
        train_acc = train_acc_metric.result()
        print('Training acc over epoch: %s' % (float(train_acc),))
        tf.summary.scalar('train/acc', train_acc, step=epoch)
        # Reset training metrics at the end of each epoch
        train_acc_metric.reset_states()

        # Run a validation loop at the end of each epoch.
        for x_batch_val, y_batch_val in val_dataset:
            val_logits = model(x_batch_val)
            # Update val metrics
            val_acc_metric(y_batch_val, val_logits)
        val_acc = val_acc_metric.result()
        val_acc_metric.reset_states()
        print('Validation acc: %s' % (float(val_acc),))
        tf.summary.scalar('val/acc', val_acc, step=epoch)