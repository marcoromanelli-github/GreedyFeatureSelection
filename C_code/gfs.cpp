//
// Created by Marco Romanelli on 11/2/20.
//

#include "gfs.h"

using namespace gfs_manager_space;


gfsManager::gfsManager(vector<vector<string>> dataset, vector<string> labels_input, string strategy_arg) {
    this->labels = labels_input;
    int rows = dataset.size();
    if (rows == 0) {
        throw std::invalid_argument("no samples available");
    } else {
        for (int col_idx = 0; col_idx < dataset[0].size(); ++col_idx) {
            vector<string> column;
            for (int i = 0; i < rows; ++i) {
                column.push_back(dataset.at(i).at(col_idx));
            }
            this->F.insert(make_pair(col_idx, column));

        }
    }

    if (isInList(strategy_arg, this->available_strategies)) {
        this->strategy = strategy_arg;
    } else {
        throw std::invalid_argument("required strategy not known");
    }
}

gfsManager::~gfsManager() {
}

vector<string> gfsManager::getFVal(int key) {
    return this->F[key];
}

vector<int> gfsManager::greedyAlgorithm(int feat_card) {
    for (int step = 0; step, feat_card; ++step) {   //  feature selection step: step

    }
    return this->S_index;
}

void gfsManager::gfsPickNextFeature() {
    map<int, vector<string>>::iterator it;
    map<int, float> entropy_val;  // contains all the entropy values at a certain step: the argmax corresponds to
    // the next chosen feature unequivocally identified by the int id

    if (this->S.empty()) {   //  first step
        for (it = this->F.begin(); it != this->F.end(); it++) {
            vector<string> S_t = newFeature(this->S, it->second); // contains all the new possible symbols for a
            // given new feature
            entropy_val.insert(make_pair(it->first, computeEntropy(S_t, this->labels)));//
        }
    } else {   //  next steps

    }
}

vector<string> gfsManager::newFeature(vector<string> S_base, vector<string> feature_array) {
    vector<string> new_feature;

    if (S_base.empty()) {
        return feature_array;
    } else {
        if ((S_base.size() != feature_array.size()) || (feature_array.empty())) {
            throw std::invalid_argument("arrays' dimensions mismatch");
        } else {
            for (int i = 0; i < S_base.size(); ++i) {
                new_feature.push_back(S_base.at(i) + "_" + feature_array.at(i));
            }
        }
        return new_feature;
    }
}

float gfsManager::computeEntropy(vector<string> feature_array, vector<string> labels_array) {
    if (this->strategy == "renyi") {
        return 1.0;
    } else if (this->strategy == "shannon") {
        return 2.0;
    } else {
        throw std::invalid_argument("required strategy not known");
    }
}

template<typename T>
static bool isInList(T element, vector<T> the_list) {
    if (!the_list.empty()) {
        if (std::find(the_list.begin(), the_list.end(), element) != the_list.end()) {
            return true;
        } else {
            return false;
        }
    } else {
        return false;
    }
}
