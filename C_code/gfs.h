//
// Created by Marco Romanelli on 11/2/20.
//

#ifndef GREEDYFEATURESELECTION_GFS_H
#define GREEDYFEATURESELECTION_GFS_H

#include <map>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>

using namespace std;

namespace gfs_manager_space {
    class gfsManager {
    public:
        gfsManager(vector<vector<string>> dataset, vector<string> labels_input, string strategy_arg);

        ~gfsManager();

        vector<int> greedyAlgorithm(int feat_card);

        vector<string> getFVal(int key);

        void gfsPickNextFeature();

        float computeEntropy(vector<string> feature_array, vector<string> labels_array);

    private:
        int rows;
        vector<string> available_strategies = {"renyi", "shannon"};
        vector<string> labels;
        vector<string> S;   // contains symbols corresponding to the features chosen so far
        vector<int> S_index;   // contains symbols corresponding to the features chosen so far
        map<int, vector<string>> F;  // map which contains all the indices int and the feature columns which have not
        // been selected yet
        string strategy;

    };

}

template<typename T>
static bool isInList(T element, vector<T> the_list);
static vector<string> newFeature(vector<string> S_base, vector<string> feature_array);
static int getIndexMaxValueMap(map<int, float> map_obj);
#endif //GREEDYFEATURESELECTION_GFS_H
