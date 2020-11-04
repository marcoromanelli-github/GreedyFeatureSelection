//
// Created by Marco Romanelli on 11/2/20.
//

#include "gfs.h"

using namespace gfs_manager_space;

gfsManager::gfsManager(vector<vector<int> > dataset, vector<int> labels_input, string strategy_arg) {
    for (int i = 0; i < labels_input.size(); ++i) {
        this->labels.push_back(to_string(labels_input.at(i)));
    }

    this->available_strategies.push_back("shannon");
    this->available_strategies.push_back("renyi");

    this->labels_set = vectorToSet(this->labels);
    this->rows = dataset.size();
    if (!isMatrix(dataset)) {
        throw std::invalid_argument("no samples available or columns number mismatch");
    } else {
        int cols = dataset[0].size();
        for (int col_idx = 0; col_idx < cols; ++col_idx) {
            vector<string> column;
            column.reserve(cols);
            for (int i = 0; i < this->rows; ++i) {
                column.push_back(to_string(dataset.at(i).at(col_idx)));
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
    if (feat_card < 1 || feat_card >= this->rows) {
        throw std::invalid_argument("required number of feature to select must be an integer greater than 0 and "
                                    "less than the total number of available features");
    }
    for (int step = 0; step < feat_card; ++step) {   //  feature selection step: step
        gfsPickNextFeature();
    }
    return this->S_index;
}

void gfsManager::gfsPickNextFeature() {
    map<int, vector<string> >::iterator it;
    map<int, float> entropy_val;  // contains all the entropy values at a certain step: the argmax corresponds to
    // the next chosen feature unequivocally identified by the int id

    for (it = this->F.begin(); it != this->F.end(); it++) {
        vector<string> S_t = newFeature(this->S, it->second); // contains all the new possible symbols for a
        // given new feature
        entropy_val.insert(
                make_pair(it->first,
                          computeEntropy(S_t,
                                         this->labels,
                                         vectorToSet(S_t),
                                         this->labels_set)));
    }
    int max_entropy_index = getIndexMinValueMap(entropy_val);
    this->S_index.push_back(max_entropy_index);
    this->S = newFeature(this->S, F[max_entropy_index]);
    this->F.erase(max_entropy_index);
}

float gfsManager::computeEntropy(vector<string> feature_array, vector<string> labels_array,
                                 set<string> feature_array_set, set<string> labels_array_set) {
    if (this->strategy == "renyi") {
        return renyiMinEntropy(feature_array, labels_array, feature_array_set, labels_array_set);
    } else if (this->strategy == "shannon") {
        return shannonEntropy(feature_array, labels_array, feature_array_set, labels_array_set);
    } else {
        throw std::invalid_argument("required strategy not known");
    }
}

map<int, vector<string> > gfsManager::getF() {
    return this->F;
}

//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
//%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

bool isInList(string element, vector<string> the_list) {
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

int getIndexMinValueMap(map<int, float> map_obj) {
    float current_min = numeric_limits<float>::infinity();
    int current_min_id = -1;

    map<int, float>::iterator it;
    for (it = map_obj.begin(); it != map_obj.end(); it++) {
        if (it->second < current_min) {
            current_min = it->second;
            current_min_id = it->first;
        }
    }
    return current_min_id;
}

vector<string> newFeature(vector<string> S_base, vector<string> feature_array) {
    vector<string> new_feature;
    int feature_array_size = feature_array.size();
    int S_base_size = S_base.size();
    new_feature.reserve(feature_array_size);

    if (S_base.empty()) {
        return feature_array;
    } else {
        if ((S_base_size != feature_array_size) || (feature_array.empty())) {
            throw std::invalid_argument("arrays' dimensions mismatch");
        } else {
            for (int i = 0; i < S_base_size; ++i) {
                new_feature.push_back(S_base.at(i) + "_" + feature_array.at(i));
            }
        }
        return new_feature;
    }
}

map<string, float> computeProb(vector<string> vec) {
    float number_of_samples = vec.size();

    map<string, int> joint_freq;

    for (int i = 0; i < vec.size(); ++i) {
        string current_element = vec.at(i);
        if (joint_freq.find(current_element) == joint_freq.end()) {
            joint_freq.insert(make_pair(current_element, 1));
        } else {
            joint_freq[current_element] += 1;
        }
    }

    map<string, float> joint_prob;
    map<string, int>::iterator it;
    for (it = joint_freq.begin(); it != joint_freq.end(); it++) {
        joint_prob[it->first] = it->second / number_of_samples;
    }

    return joint_prob;
}

map<string, float> computeJointProb(vector<string> vec_0, vector<string> vec_1) {
    vector<string> vec_tmp = newFeature(vec_0, vec_1);
    return computeProb(vec_tmp);
}

float shannonEntropy(vector<string> Y, vector<string> X, set<string> Y_set, set<string> X_set) {
    map<string, float> P_XjointY_map = computeJointProb(X, Y);
    map<string, float> P_Y_map = computeProb(Y);
//    printMap(P_XjointY_map);
//    cout << "///";
//    printMap(P_Y_map);
//    cout << "///";
    float sum_ext = 0;
    for (string x_el : X_set) {
        string x = x_el;

        float sum_int = 0;

        for (string y_el : Y_set) {
            string y = y_el;
            float p_y = P_Y_map[y];
            string string_tmp = x + "_" + y;
            if (isKey(P_XjointY_map, string_tmp)) {
                float p_xjointy = P_XjointY_map[string_tmp];
//                cout << string_tmp << "  p_xjointy  " << p_xjointy << " p_y  " << p_y << endl;
                sum_int += p_xjointy * log2(p_xjointy / p_y);
            }
        }

        sum_ext += sum_int;
    }
//    cout << -sum_ext << endl << endl;
    return -sum_ext;
}

float renyiMinEntropy(vector<string> Y, vector<string> X, set<string> Y_set, set<string> X_set) {
    map<string, float> P_XjointY_map = computeJointProb(X, Y);
    float sum_ext = 0;
    for (string y_el : Y_set) {
        string y = y_el;

        float max_int = -numeric_limits<float>::infinity();;

        for (string x_el : X_set) {
            string x = x_el;
            string string_tmp = x + "_" + y;
            if (isKey(P_XjointY_map, string_tmp)) {
                float p_xjointy = P_XjointY_map[string_tmp];
                if (p_xjointy > max_int) {
                    max_int = p_xjointy;
                }
            }
        }
        sum_ext += max_int;
    }
    return -log2(sum_ext);
}

