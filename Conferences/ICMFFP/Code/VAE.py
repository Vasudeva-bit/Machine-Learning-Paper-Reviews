class VAE(tf.keras.Model):
    def __init__(self,
                latent_dim=100,
                n_filters=12,
                ):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.n_filters = n_filters

        self.encoder = None
        self.decoder = None

    def _sampling(self, z_mean, z_logsigma):
        batch, latent_dim = z_mean.shape
        epsilon = tf.random.normal(shape=(batch, latent_dim))
        z = z_mean + (tf.math.exp(z_logsigma)*epsilon)
        return z

    def _decoder(self):
        Conv2DTranspose = functools.partial(
            tf.keras.layers.Conv2DTranspose,
            padding='same',
            activation='relu')
        Dense = functools.partial(tf.keras.layers.Dense, activation='relu')
        Reshape = tf.keras.layers.Reshape

        model = tf.keras.Sequential([
            # Transform to pre-convolutional generation 4x4 feature maps (with 6N occurances)
            Dense(units=14*14*6*self.n_filters),
            Reshape(target_shape=(14, 14, 6*self.n_filters)),
            # Upscaling convolutions (inverse of encoder)
            Conv2DTranspose(filters=4*self.n_filters, kernel_size=3,  strides=2),
            Conv2DTranspose(filters=2*self.n_filters, kernel_size=3,  strides=2),
            Conv2DTranspose(filters=1*self.n_filters, kernel_size=5,  strides=2),
            Conv2DTranspose(filters=3, kernel_size=5,  strides=2),
        ])
        return model

    def _encoder(self, n_outputs=1):
        Conv2D = functools.partial(
            tf.keras.layers.Conv2D, padding='same', activation='relu')
        BatchNormalization = tf.keras.layers.BatchNormalization
        Flatten = tf.keras.layers.Flatten
        Dense = functools.partial(tf.keras.layers.Dense, activation='relu')

        model = tf.keras.Sequential([
            Conv2D(filters=1*self.n_filters, kernel_size=5,  strides=2),
            BatchNormalization(),

            Conv2D(filters=2*self.n_filters, kernel_size=5,  strides=2),
            BatchNormalization(),

            Conv2D(filters=4*self.n_filters, kernel_size=3,  strides=2),
            BatchNormalization(),

            Conv2D(filters=6*self.n_filters, kernel_size=3,  strides=2),
            BatchNormalization(),

            Flatten(),
            Dense(512),
            Dense(n_outputs, activation=None),
        ])
        return model

    def encode(self, x):
        num_encoder_dims = 2*self.latent_dim # +1
        self.encoder = self._encoder(num_encoder_dims)
        encoder_output = self.encoder(x)
        z_mean = encoder_output[:, 0:self.latent_dim]
        z_logsigma = encoder_output[:, self.latent_dim:]
        return z_mean, z_logsigma

    def reparameterize(self, z_mean, z_logsigma):
        z = self._sampling(z_mean, z_logsigma)
        return z

    def decode(self, z):
        self.decoder = self._decoder()
        reconstruction = self.decoder(z)
        return reconstruction

    def call(self, x):
        z_mean, z_logsigma = self.encode(x)
        z = self.reparameterize(z_mean, z_logsigma)
        recon = self.decode(z)
        return z_mean, z_logsigma, recon

    def vae_loss_function(self, x, x_recon, mu, logsigma, kl_weight=0.0005):
        latent_loss = tf.reduce_sum((tf.math.exp(logsigma))+((mu**2)-1-logsigma), axis=0)
        reconstruction_loss = tf.math.reduce_mean(tf.math.abs(x - x_recon))
        vae_loss = (kl_weight*latent_loss)+reconstruction_loss
        return vae_loss

    @tf.function
    def vae_training_step(self, x):
        with tf.GradientTape() as tape:
            z_mean, z_logsigma, x_recon = vae.call(x)
            loss = self.vae_loss_function(x, x_recon, z_mean, z_logsigma)
        grads = tape.gradient(loss, vae.trainable_variables)  # TODO
        optimizer_vae.apply_gradients(zip(grads, vae.trainable_variables))
        return loss