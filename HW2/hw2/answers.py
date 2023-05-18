r"""
Use this module to write your answers to the questions in the notebook.

Note: Inside the answer strings you can use Markdown format and also LaTeX
math (delimited with $$).
"""

# ==============
# Part 2 answers


def part2_overfit_hp():
    wstd, lr, reg = 0, 0, 0
    # TODO: Tweak the hyperparameters until you overfit the small dataset.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_optim_hp(opt_name):
    wstd, lr, reg, = 0, 0, 0

    # TODO: Tweak the hyperparameters to get the best results you can.
    # You may want to use different hyperparameters for each optimizer.
    # ====== YOUR CODE: ======
    if opt_name == 'vanilla':
        raise NotImplementedError()
    if opt_name == 'momentum':
        raise NotImplementedError()
    if opt_name == 'rmsprop':
        raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr, reg=reg)


def part2_dropout_hp():
    wstd, lr, = 0, 0
    # TODO: Tweak the hyperparameters to get the model to overfit without
    # dropout.
    # ====== YOUR CODE: ======
    raise NotImplementedError()
    # ========================
    return dict(wstd=wstd, lr=lr)


part2_q1 = r"""

"""

part2_q2 = r"""
"""
# ==============

# ==============
# Part 3 answers

part3_q1 = r"""
We tested hidden dims of 100 and 1024, both were pretty similar.

For L2, L4 I tested with pool every 2, for L8 I tried pool every 2 or 4, for L16 I tried pool every 4 or 8.

L4 provided with the best test accuracy of 0.7, L2 was a little bit behind.

L8 and L16 mostly provided test accuracy same as random (10%) meaning they didn't manage to train at all.

except for K64_L8 when pooling every 4 rounds, which did manage to train to a test accuracy of 0.6 .

We think L4 was better than L2 because the network was richer, but L8 and L16 were worse.
This can happen because of the vanishing gradients problem or overfitting.

To resolve this:

we can add skip or residual connections to help the gradient flow.

we can add batch normalization to help the gradient flow.

Pooling less can partially (as we saw with L8 but not L16) help preserve more details and allow the gradient to flow back more accurately.

Larger networks require more data, maybe a larger dataset can improve accuracy.
"""

part3_q2 = r"""
for a thinner network like L2, adding too much filters reduced performance, K32 was best with 0.67 test accuracy. more filters gave worse results.

the richer L4 network liked more filters, and K256 was best with 0.72 test accuracy - this is better than it performed in q1. less filters gave worse results.

L8 was not trainable until we reduced pooling to every 4, and then it reached 0.63 test accuracy with K64 and K128. other number of filters remained untrainable. 
"""

part3_q3 = r"""
When choosing hidden dims = 1024, only L1 managed to train.
When choosing hidden dims = 100, also L2 managed to train.

We chose to pool every equal to the number of layers.

L2 was best with 0.76 test accuracy, and L1 was close with 0.74.

L3, L4 didn't train for any parameters I tested.

The results for L1, L2 are better than in q1
"""


part3_q4 = r"""
We added batch normalization after every convolutional layer and a skip connection once before every pooling, this helped eliminate the vanishing gradient.
In the skip connection we also changed the kernel size and stride to 1.
We tried adding dropout but this didn't improve the accuracy (We are guessing because overfitting wasnt the case) so we decided to remove it

L4 was best with test accuracy of 0.8, followed by L3 with test accuracy of 0.79.
They improved the most, because previously they were completely untrainable.
This happened because the normalization and skip connections helped eliminate the vanishing gradient problem
L2 and L1 didn't improve much and remained with around the same accuracy as in q3.
"""
