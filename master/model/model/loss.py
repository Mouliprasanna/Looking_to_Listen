import tensorflow as tf

def audio_discriminate_loss(gamma=0.1, people_num=2):
    def loss_func(S_true, S_pred, gamma=gamma, people_num=people_num):
        total_loss = 0
        for i in range(people_num):
            total_loss += tf.reduce_sum(tf.square(S_true[:, :, :, i] - S_pred[:, :, :, i]))
            for j in range(people_num):
                if i != j:
                    total_loss -= gamma * tf.reduce_sum(tf.square(S_true[:, :, :, i] - S_pred[:, :, :, j]))

        loss = total_loss / (people_num * 298 * 257 * 2)
        return loss
    return loss_func


def audio_discriminate_loss2(gamma=0.1, beta=2 * 0.1, people_num=2):
    def loss_func(S_true, S_pred, gamma=gamma, beta=beta, people_num=people_num):
        sum_mtr = tf.zeros_like(S_true[:, :, :, :, 0])
        for i in range(people_num):
            sum_mtr += tf.square(S_true[:, :, :, :, i] - S_pred[:, :, :, :, i])
            for j in range(people_num):
                if i != j:
                    sum_mtr -= gamma * tf.square(S_true[:, :, :, :, i] - S_pred[:, :, :, :, j])

        # Uncomment and modify if needed for the additional conditions
        # for i in range(people_num):
        #     for j in range(i + 1, people_num):
        #         sum_mtr -= beta * tf.square(S_pred[:, :, :, i] - S_pred[:, :, :, j])
        #         sum_mtr += beta * tf.square(S_true[:, :, :, i] - S_true[:, :, :, j])

        # Correctly flatten and reduce mean
        loss = tf.reduce_mean(tf.reshape(sum_mtr, [-1]))

        return loss
    return loss_func
