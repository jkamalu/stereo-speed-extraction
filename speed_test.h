#include <map>
#include <vector>
#include <tuple>
#include <string>
#include "speed_extractor.h"

using namespace std;

class SpeedTest {

    private:
    
        // typedefs
        struct Position {
            float x, y, z;
        };
    
        struct Rotation {
            float x, y, z;
        };
    
        typedef map<string, vector<tuple<int, Position, Rotation> > > Timesteps;
    
        typedef map<string, vector<float> > Speeds;
    
        // class variables
        string path;
    
        float speed;
    
        Timesteps timesteps;
    
        Speeds speeds;
    
        SpeedExtractor speedExtractor;

        // functions
        float getSpeed(string l0, string l1, string r0, string r1, int dt);
    
        inline string getFilename(string config, int time) {
            return path + "/" + config + "/" + to_string(time);
        };

    public:

        // constructors
        SpeedTest(string path, float speed);

        // functions
        vector<string> vectorize(string line);
    
        void calculateSpeeds();
    
        void speedStats();

};
