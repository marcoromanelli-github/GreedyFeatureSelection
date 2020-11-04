//
// Created by Marco Romanelli on 11/2/20.
//

#ifndef GREEDYFEATURESELECTION_GFS_H
#define GREEDYFEATURESELECTION_GFS_H


#include <map>
#include <set>
#include <cmath>
#include <vector>
#include <iostream>
#include <algorithm>
#include <stdexcept>

using namespace std;

namespace gfs_manager_space {
    class gfsManager {
    public:
        gfsManager(vector< vector< int > > dataset, vector<int> labels_input, string strategy_arg);

        ~gfsManager();

        vector<int> greedyAlgorithm(int feat_card);

        vector<string> getFVal(int key);

        void gfsPickNextFeature();

        float computeEntropy(vector<string> feature_array, vector<string> labels_array, set<string> feature_array_set,
                             set<string> labels_array_set);

        map<int, vector<string> > getF();

    private:
        int rows;
        vector<string> available_strategies;
        vector<string> labels;
        set<string> labels_set;
        vector<string> S;   // contains symbols corresponding to the features chosen so far
        vector<int> S_index;   // contains symbols corresponding to the features chosen so far
        map<int, vector<string> > F;  // map which contains all the indices int and the feature columns which have not
        // been selected yet
        string strategy;

    };
}

bool isInList(string element, vector<string> the_list);

template<typename T0, typename T1>
void printMap(map<T0, T1> map_0) {
    typename map<T0, T1>::iterator it;
    for (it = map_0.begin(); it != map_0.end(); ++it) {
        cout << it->first << " => " << it->second << endl;
    }
}

int getIndexMinValueMap(map<int, float> map_obj);

vector<string> newFeature(vector<string> S_base, vector<string> feature_array);

map<string, float> computeProb(vector<string> vec);

map<string, float> computeJointProb(vector<string> vec_0, vector<string> vec_1);

float shannonEntropy(vector<string> Y, vector<string> X, set<string> Y_set, set<string> X_set);

float renyiMinEntropy(vector<string> Y, vector<string> X, set<string> Y_set, set<string> X_set);

template<typename T0, typename T1>
bool isKey(map<T0, T1> m, T0 possible_key) {
    return m.find(possible_key) != m.end();
}

template<typename T>
set<T> vectorToSet(vector<T> v) {
    set<T> s;
    for (T x : v) {
        s.insert(x);
    }
    return s;
}

template<typename T>
void printSet(set<T> s) {
    cout << "Set: ";
    for (T x : s) {
        cout << x << " ";
    }
    cout << endl;
}

template<typename T>
bool isMatrix(vector< vector<T> > vec) {
    int rows = vec.size();
    if (rows < 0) {
        return false;
    }
    int cols = vec[0].size();
    for (int i = 1; i < rows; ++i) {
        if (cols != vec[i].size()) {
            return false;
        }
    }
    return true;
}

template<typename T>
void printArray(vector<T> vec) {
    for (int i = 0; i < vec.size(); i++) {
        cout << vec[i] << "\t";
    }
    cout << "\n" << endl;
}

template<typename T>
void printMatrix(vector< vector<T> > vec) {
    for (int i = 0; i < vec.size(); i++) {
        for (int j = 0; j < vec[i].size(); j++) {
            cout << vec[i][j] << "\t";
            if (j == vec[i].size() - 1) {
                cout << endl;
            }
        }
    }
    cout << "\n" << endl;
}

#endif //GREEDYFEATURESELECTION_GFS_H
