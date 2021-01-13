from csv_reader import read_csv_file

from stats_calculator import calc_sample_mode, calc_sample_median, calc_linear_reg, calc_correlation_coefficiant, \
    calc_sample_mean, calc_std_dev
import matplotlib.pyplot as plt
import sys
import numpy as np
import tensorflow as tf

def main():
    if len(sys.argv) > 2:
        filename = sys.argv[2]
        set1, set2, set3 = read_csv_file(filename)
        print("$ python swen-102 AI project V0.4, datafile = " + filename + "\n")
    else:
        set1 = np.linspace(0, 20, 50)
        set2 = np.random.randn() * set1 + np.random.randn() * .09 + 2
        set3 = None
        print("$ python swen-102 AI project V0.4, datafile = None\n")

    mean1 = calc_sample_mean(set1)
    mean2 = calc_sample_mean(set2)
    median1 = calc_sample_median(set1)
    median2 = calc_sample_median(set2)
    mode1 = calc_sample_mode(set1)
    mode2 = calc_sample_mode(set2)
    stddev1 = calc_std_dev(set1, mean1)
    stddev2 = calc_std_dev(set2, mean2)
    r = calc_correlation_coefficiant(stddev1, stddev2, mean1, mean2, set1, set2)

    if set3 is not None:
        mean3 = calc_sample_mean(set3)
        median3 = calc_sample_median(set3)
        mode3 = calc_sample_mode(set3)
        stddev3 = calc_std_dev(set3, mean3)
        r2 = calc_correlation_coefficiant(stddev1, stddev3, mean1, mean3, set1, set3)
        r3 = calc_correlation_coefficiant(stddev2, stddev3, mean2, mean3, set2, set3)
        #3 variable correlation equation
        r = format(((r2**2)+(r3**2)-2.0*(r2*r3*r))/((1-(r**2))**.5), '.2f')

    if set3 is None:
        slope1, y_intercept = calc_linear_reg(stddev1, stddev2, mean1, mean2, r, False)

        print("RESULTS: \nMEANS: " + str(mean1) + " and " + str(mean2) + "\nMEIDANS: " + str(median1) + " and " + str(
            median2) + "\nMODES: " + str(mode1) + " and " + str(mode2) + "\nSTANDARD DEVIATIONS: " + str(
            stddev1) + " and " + str(stddev2) + "\nLINEAR REGRESSION EQUATION:  y = " + str(slope1) + "x + " + str(
            y_intercept) + "\nr = " + str(r))

        plt.plot(set1, set2, 'ro', label="Given data")

        y_set = []
        for value in set1:
            y_set.append(value * slope1 + y_intercept)

        plt.plot(set1, y_set, label="Regression Line")
        plt.legend()
        plt.show()

    else:

        slope1, slope2, intercept = calc_linear_reg(stddev1, stddev2, mean1, mean2, r, True, set1, set2, set3, mean3)

        print("RESULTS: \nMEANS: " + str(mean1) + " and " + str(mean2) + " and " + str(mean3) + "\nMEDIANS: " + str(median1) + " and " + str(
            median2) +  " and " + str(median3) + "\nMODES: " + str(mode1) + " and " + str(mode2) + " and " + str(mode3) + "\nSTANDARD DEVIATIONS: " + str(
            stddev1) + " and " + str(stddev2) + " and " + str(stddev3) + "\nLINEAR REGRESSION EQUATION:  z = " + str(slope1) + "x + " + str(slope2) + "y + " + str(
            intercept) + "\nr = " + str(r))

    # Setup training parameters

    n_trials = int(sys.argv[1])

    #Collect training variables (First 80% of data)
    train_x = np.asarray(set1[:int(len(set1) * .8)])
    train_y = np.asarray(set2[:int(len(set2) * .8)])

    #Testing data (other 20% of data)
    test_x = np.asarray(set1[int(len(set1) * .8):])
    test_y = np.asarray(set2[int(len(set2) * .8):])

    X = tf.placeholder("float")
    Y = tf.placeholder("float")

    W = tf.Variable(np.random.randn(), name="Weight")
    B = tf.Variable(np.random.randn(), name="bias")


    if set3 is not None:
        train_z = np.asarray(set3[:int(len(set3) * .8)])
        test_z = np.asarray(set3[int(len(set3) * .8):])
        Z = tf.placeholder("float")
        V = tf.Variable(np.random.randn(), name="Weight2")
        pred = tf.add(tf.add(tf.multiply(X, W), tf.multiply(Y, V)), B)
        loss = tf.reduce_sum(tf.pow(pred - Z, 2)) / (2 * len(train_z))

    else:
        # Setup vaiables and establish regression line equation
        # Y = W*X + B
        pred = tf.add(tf.multiply(X, W), B)

        #Calculate Mean Squared Error (how far the data is from the most recent regression line)
        loss = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2*len(train_x))

    #Optimize at .01 learning rate and learn by reducing MSE
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(.01).minimize(loss)

    init = tf.compat.v1.global_variables_initializer()

    # train the model through the session
    with tf.compat.v1.Session() as session:

        session.run(init)

        check = False

        if set3 is None:
            for epoch in range(n_trials):
                if check is False:
                    for (x, y) in zip(train_x, train_y):
                        session.run(optimizer, feed_dict={X: x, Y: y})

                    b = float(format(session.run(B), '.2f'))
                    w = float(format(session.run(W), '.2f'))

                    if (epoch + 1) % 100 == 0:
                        c = session.run(loss, feed_dict={X: train_x, Y: train_y})
                        print("Epoch " + str((epoch + 1)) + " Error = ", "{:.3}".format(c), "W = ", w,
                              " B = ", b)

                    if w == slope1 and b == intercept:
                        print("Solved!!!")
                        check = True

        else:
            for epoch in range(n_trials):
                if check is False:
                    for (x, y, z) in zip(train_x, train_y, train_z):
                        session.run(optimizer, feed_dict={X: x, Y: y, Z: z})

                    b = float(format(session.run(B), '.2f'))
                    w = float(format(session.run(W), '.2f'))
                    v = float(format(session.run(V), '.2f'))

                    if (epoch + 1) % 100 == 0:
                        c = session.run(loss, feed_dict={X: train_x, Y: train_y, Z: train_z})
                        print("Epoch " + str((epoch + 1)) + " Error = ", "{:.3}".format(c), "W = ", w,
                              " V = ", v, " B = ", b)

                    if w == slope1 and v == slope2 and b == intercept:
                        print("Solved!!!")
                        check = True

        print("=====================\nOptimizations Finished")

        if set3 is not None:
            training_error = session.run(loss, feed_dict={X: train_x, Y: train_y, Z: train_z})
            if check:
                print("Training Error = ", "{:.4}".format(training_error), " W = ", w, " V = ", v, " B = ", b,
                      "\nFORMULA: "+str(w)+"x + "+str(b)+"\nSolution found after "+str(epoch)+" Epochs")
            else:
                w = float(format(session.run(W), '.2f'))
                v = float(format(session.run(V), '.2f'))
                b = float(format(session.run(B), '.2f'))
                print("Training Error = ", "{:.4}".format(training_error), " W = ", w, " V = ", v, " B = ",
                      b, "\nFORMULA: "+str(w)+"x + "+str(b)+"\nSolution found after "+str(n_trials)+" Epochs")
        else:
            training_error = session.run(loss, feed_dict={X: train_x, Y: train_y})
            if check:
                print("Training Error = ", "{:.4}".format(training_error), " W = ", w, " B = ", b,
                      "\nFORMULA: " + str(w) + "x + " + str(b) + "\nSolution found after " + str(epoch) + " Epochs")
            else:
                w = float(format(session.run(W), '.2f'))
                b = float(format(session.run(B), '.2f'))
                print("Training Error = ", "{:.4}".format(training_error), " W = ", w, " B = ", b,
                      "\nFORMULA: " + str(w) + "x + " + str(b) + "\nSolution found after " + str(n_trials) + " Epochs")

            plt.plot(train_x, train_y, 'ro', label="Original data")
            plt.plot(train_x, session.run(W) * train_x + session.run(B), label="Fitted Line")
            plt.legend()
            plt.show()

#Test the results with remaining data

        print("Now Testing...")
        if set3 is not None:
            test_error = session.run(tf.reduce_sum(tf.pow(pred - Z, 2))/(2*test_z.shape[0]), feed_dict={X: test_x, Y: test_y, Z: test_z})
        else:
            test_error = session.run(tf.reduce_sum(tf.pow(pred-Y, 2)/(2*test_x.shape[0])), feed_dict={X: test_x, Y: test_y})
            plt.plot(test_x, test_y, 'bo', label="Test data")
            plt.plot(train_x, session.run(W) * train_x + session.run(B), label="Fitted Line")
            plt.legend()
            plt.show()

        print("Test Error = ", test_error)
        print("Abs Square Mean Loss Difference = ", abs(test_error-training_error))

if __name__ == '__main__':
    main()
