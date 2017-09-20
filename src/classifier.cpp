#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>
#include "classifier.h"

/**
 * Initializes GNB
 */

GNB::GNB() {
    p_left = 0.0;
    p_keep = 1.0;
    p_right = 0.0;
}

GNB::~GNB() {}

void GNB::train(vector<vector<double>> data, vector<string> labels)
{
	/*
		Trains the classifier with N data points and labels.

		INPUTS

		data - array of N observations
		  - Each observation is a tuple with 4 values: s, d, 
		    s_dot and d_dot.
		  - Example : [
			  	[3.5, 0.1, 5.9, -0.02],
			  	[8.0, -0.3, 3.0, 2.2],
			  	...
		  	]

		labels - array of N labels
		  - Each label is one of "left", "keep", or "right".
	*/

    //
    // Statistics for p_left, p_keep and p_right
    // p_total = p_left + p_keep + p_right = 1
    // p_left = left_cnt/total_cnt and so on.
    //
    // Or shall we assume 33.3% of all three cases?
    //

    // Features:
    // Use  d_in_lane = d % lane_width,
    //      d_dot = delta_d/delta_t
    //      s_dot = delta_s/delta_t
    // Use Gaussian distribution to emulate the probability
    //      need to get u and sigma of each feature
    //      u = average over all same labelled data
    //      sigma^2 = sum((x_i - u)^2)/N

    // training items count
    unsigned int p_total_cnt = 0;
    unsigned int p_left_cnt = 0;
    unsigned int p_keep_cnt = 0;
    unsigned int p_right_cnt = 0;

    // -------------
    // average
    // -------------
    double left_d_offset_acc = 0.0;
    double keep_d_offset_acc = 0.0;
    double right_d_offset_acc = 0.0;
    double left_d_offset_ave = 0.0;
    double keep_d_offset_ave = 0.0;
    double right_d_offset_ave = 0.0;

    double left_s_dot_acc = 0.0;
    double keep_s_dot_acc = 0.0;
    double right_s_dot_acc = 0.0;
    double left_s_dot_ave = 0.0;
    double keep_s_dot_ave = 0.0;
    double right_s_dot_ave = 0.0;

    double left_d_dot_acc = 0.0;
    double keep_d_dot_acc = 0.0;
    double right_d_dot_acc = 0.0;
    double left_d_dot_ave = 0.0;
    double keep_d_dot_ave = 0.0;
    double right_d_dot_ave = 0.0;

    for (int i=0; i<labels.capacity(); i++)
    {
        string tmp = labels[i];
        double tmp2 = fmod(data[i][1], lane_width);
        double tmp3 = lane_width - tmp2;
        tmp2 = (tmp2 > tmp3) ? tmp3 : tmp2;

        double tmp4 = data[i][2];
        double tmp5 = data[i][3];

        if (tmp == "left") {
            p_left_cnt++;
            left_d_offset_acc += tmp2;
            left_s_dot_acc += tmp4;
            left_d_dot_acc += tmp5;
        } else if (tmp == "keep") {
            p_keep_cnt++;
            keep_d_offset_acc += tmp2;
            keep_s_dot_acc += tmp4;
            keep_d_dot_acc += tmp5;
        } else if (tmp == "right") {
            p_right_cnt++;
            right_d_offset_acc += tmp2;
            right_s_dot_acc += tmp4;
            right_d_dot_acc += tmp5;
        }

    }

    p_total_cnt = p_left_cnt + p_keep_cnt + p_right_cnt;

    left_d_offset_ave = left_d_offset_acc / ((double) p_left_cnt);
    keep_d_offset_ave = keep_d_offset_acc / ((double) p_keep_cnt);
    right_d_offset_ave = right_d_offset_acc / ((double) p_right_cnt);

    left_s_dot_ave = left_s_dot_acc / ((double) p_left_cnt);
    keep_s_dot_ave = keep_s_dot_acc / ((double) p_keep_cnt);
    right_s_dot_ave = right_s_dot_acc / ((double) p_right_cnt);

    left_d_dot_ave = left_d_dot_acc / ((double) p_left_cnt);
    keep_d_dot_ave = keep_d_dot_acc / ((double) p_keep_cnt);
    right_d_dot_ave = right_d_dot_acc / ((double) p_right_cnt);

//    cout << "p_total_cnt=" << p_total_cnt << ", p_left_cnt=" << p_left_cnt
//         << ", p_keep_cnt=" << p_keep_cnt << ", p_right_cnt=" << p_right_cnt
//         << endl;

    // Update behavior occurring probability
    p_left = ((double) p_left_cnt) / ((double) p_total_cnt);
    p_keep = ((double) p_keep_cnt) / ((double) p_total_cnt);
    p_right = ((double) p_right_cnt) / ((double) p_total_cnt);

    // -------------
    // deviation
    // -------------
    double left_d_offset_dev_acc = 0.0;
    double keep_d_offset_dev_acc = 0.0;
    double right_d_offset_dev_acc = 0.0;
    double left_d_offset_dev = 0.0;
    double keep_d_offset_dev = 0.0;
    double right_d_offset_dev = 0.0;

    double left_s_dot_dev_acc = 0.0;
    double keep_s_dot_dev_acc = 0.0;
    double right_s_dot_dev_acc = 0.0;
    double left_s_dot_dev = 0.0;
    double keep_s_dot_dev = 0.0;
    double right_s_dot_dev = 0.0;

    double left_d_dot_dev_acc = 0.0;
    double keep_d_dot_dev_acc = 0.0;
    double right_d_dot_dev_acc = 0.0;
    double left_d_dot_dev = 0.0;
    double keep_d_dot_dev = 0.0;
    double right_d_dot_dev = 0.0;

    for (int i=0; i<labels.capacity(); i++)
    {
        string tmp = labels[i];
        double tmp2 = fmod(data[i][1], lane_width);
        double tmp3 = lane_width - tmp2;
        tmp2 = (tmp2 > tmp3) ? tmp3 : tmp2;

        double tmp4 = data[i][2];
        double tmp5 = data[i][3];

        if (tmp == "left") {
            left_d_offset_dev_acc += pow((tmp2 - left_d_offset_ave), 2.0);
            left_s_dot_dev_acc += pow((tmp4 - left_s_dot_ave), 2.0);
            left_d_dot_dev_acc += pow((tmp5 - left_d_dot_ave), 2.0);
        } else if (tmp == "keep") {
            keep_d_offset_dev_acc += pow((tmp2 - keep_d_offset_ave), 2.0);;
            keep_s_dot_dev_acc += pow((tmp4 - keep_s_dot_ave), 2.0);
            keep_d_dot_dev_acc += pow((tmp5 - keep_d_dot_ave), 2.0);
        } else if (tmp == "right") {
            right_d_offset_dev_acc += pow((tmp2 - right_d_offset_ave), 2.0);;
            right_s_dot_dev_acc += pow((tmp4 - right_s_dot_ave), 2.0);
            right_d_dot_dev_acc += pow((tmp5 - right_d_dot_ave), 2.0);
        }

    }

    left_d_offset_dev = left_d_offset_dev_acc / ((double) p_left_cnt);
    keep_d_offset_dev = keep_d_offset_dev_acc / ((double) p_keep_cnt);
    right_d_offset_dev = right_d_offset_dev_acc / ((double) p_right_cnt);

    left_s_dot_dev = left_s_dot_dev_acc / ((double) p_left_cnt);
    keep_s_dot_dev = keep_s_dot_dev_acc / ((double) p_keep_cnt);
    right_s_dot_dev = right_s_dot_dev_acc / ((double) p_right_cnt);

    left_d_dot_dev = left_d_dot_dev_acc / ((double) p_left_cnt);
    keep_d_dot_dev = keep_d_dot_dev_acc / ((double) p_keep_cnt);
    right_d_dot_dev = right_d_dot_dev_acc / ((double) p_right_cnt);

    cout.precision(5);
    cout << "p_left=" << fixed << p_left << ", p_keep=" << p_keep<< ", p_right=" << p_right << endl;

    cout << "left_d_offset_ave=" << fixed << left_d_offset_ave << ", keep_d_offset_ave=" << keep_d_offset_ave<< ", right_d_offset_ave=" << right_d_offset_ave << endl;
    cout << "left_d_offset_dev=" << fixed << left_d_offset_dev
         << ", keep_d_offset_dev=" << keep_d_offset_dev
         << ", right_d_offset_dev=" << right_d_offset_dev << endl;

    cout << "left_s_dot_ave=" << fixed << left_s_dot_ave << ", keep_s_dot_ave=" << keep_s_dot_ave<< ", right_s_dot_ave=" << right_s_dot_ave << endl;
    cout << "left_s_dot_dev=" << fixed << left_s_dot_dev
         << ", keep_s_dot_dev=" << keep_s_dot_dev
         << ", right_s_dot_dev=" << right_s_dot_dev << endl;

    cout << "left_d_dot_ave=" << fixed << left_d_dot_ave << ", keep_d_dot_ave=" << keep_d_dot_ave<< ", right_d_dot_ave=" << right_d_dot_ave << endl;
    cout << "left_d_dot_dev=" << fixed << left_d_dot_dev
         << ", keep_d_dot_dev=" << keep_d_dot_dev
         << ", right_d_dot_dev=" << right_d_dot_dev << endl;
}



string GNB::predict(vector<double> sample)

{
	/*
		Once trained, this method is called and expected to return 
		a predicted behavior for the given observation.

		INPUTS

		observation - a 4 tuple with s, d, s_dot, d_dot.
		  - Example: [3.5, 0.1, 8.5, -0.2]

		OUTPUT

		A label representing the best guess of the classifier. Can
		be one of "left", "keep" or "right".
		"""

		# TODO - complete this
	*/

    //
    // Prediction can be done using Gaussian Naive Bayes method.
    // Wikipedia has a very good example of sex classification here:
    //  https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Gaussian_naive_Bayes
    // The theoretcal explanation is also good:
    //  https://en.wikipedia.org/wiki/Naive_Bayes_classifier#Parameter_estimation_and_event_models
    //

	return this->possible_labels[1];
}

