#include <iostream>
#include <fstream>
#include <cmath>
#include "speed_test.h"

using namespace std;

SpeedTest::SpeedTest(string path, float speed) {
    this->path = path;
    this->speed = speed;
    string dataPath = path + "/data.csv";
    ifstream dataFile(dataPath);
    if (dataFile.is_open()) {
        string line;
        getline(dataFile, line);
        while (getline(dataFile, line)) {
            vector<string> values = vectorize(line);
            string key = values.at(0);

            if (!(this->timesteps.count(key))) {
                this->timesteps[key] = vector<tuple<int, SpeedTest::Position, SpeedTest::Rotation > >();
            }

            SpeedTest::Position pos;
            pos.x = std::stof(values.at(2));
            pos.y = std::stof(values.at(3));
            pos.z = std::stof(values.at(4));
            SpeedTest::Rotation rot;
            rot.x = std::stof(values.at(5));
            rot.y = std::stof(values.at(6));
            rot.z = std::stof(values.at(7));

            tuple<int, SpeedTest::Position, SpeedTest::Rotation> val = std::make_tuple(std::stoi(values.at(1)), pos, rot);
            this->timesteps[key].push_back(val);
        }
        dataFile.close();
    }

}

std::vector<std::string> SpeedTest::vectorize(string line) {
    vector<string> values;
    size_t last = 0;
    size_t next = line.find(",", last);
    while (next != std::string::npos) {
        values.push_back(line.substr(last, next - last));
        last = next + 1;
        next = line.find(",", last);
    }
    values.push_back(line.substr(last));
    return values;
}

float SpeedTest::getSpeed(string left0, string left1, string right0, string right1, int dt) {
    // call to Bjoern's function
    // accepts: filenames
    // returns: euclidean distance
    float euclidean = 1.0;
    return euclidean / (dt / 1000.0);
}

// map: configuration -> vector of speeds
// e.g. map["vertical"] -> [2.9, 3, 3.1] where each elem in vector is speed between t and t+1
void SpeedTest::calculateSpeeds() {
    for (const auto &pair : this->timesteps) {
        vector<tuple<int, Position, Rotation> >::iterator curr = this->timesteps[pair.first].begin();
        vector<tuple<int, Position, Rotation> >::iterator next = this->timesteps[pair.first].begin();
        ++next;
        while (next != this->timesteps[pair.first].end()) {
            int t1 = std::get<0>(*curr);
            int t2 = std::get<0>(*next);

            string t1fname = this->getFilename(pair.first, t1);
            string t2fname = this->getFilename(pair.first, t2);
            float speed = this->getSpeed(t1fname + "-Left.png", t1fname + "-Right.png",
                                         t2fname + "-Left.png", t2fname + "-Right.png",
                                         t2 - t1);

            if (!(this->speeds.count(pair.first))) {
                this->speeds[pair.first] = vector<float>();
            }

            this->speeds[pair.first].push_back(speed);
            ++curr;
            ++next;
        }
    }
}

void SpeedTest::speedStats() {
    for (const auto &pair : this->speeds) {
        float avg_error = 0;
        float avg_speed = 0;
        for (float speed : this->speeds[pair.first]) {
            avg_error += abs(speed - this->speed);
            avg_speed += speed;
        }
        avg_error /= this->speeds.size();
        avg_speed /= this->speeds.size();
        cout << pair.first << ":\t" << "abs_error = " << avg_error << ", avg_speed = " << avg_speed << endl;
    }
}

int main (int argc, char *argv[]) {
    SpeedTest speedTest(argv[1], 3);
    speedTest.calculateSpeeds();
    speedTest.speedStats();
    return 0;
}
