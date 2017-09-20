#ifndef CLASSIFIER_H
#define CLASSIFIER_H

#include <iostream>
#include <sstream>
#include <fstream>
#include <math.h>
#include <vector>

using namespace std;

class GNB {

public:
        double lane_width = 4.0; // Lane width is 4 meters.

        // Behavior occuring probability
        double p_left, p_keep, p_right;

	vector<string> possible_labels = {"left","keep","right"};

	/**
  	* Constructor
  	*/

 	GNB();

	/**
 	* Destructor
 	*/

 	virtual ~GNB();

 	void train(vector<vector<double> > data, vector<string>  labels);

  	string predict(vector<double>);
};



#endif




