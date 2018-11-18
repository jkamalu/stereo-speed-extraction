#include <map>
#include <vector>
#include <tuple>
#include <string>

using namespace std;

class SpeedTest {

    private:
        // class variables
        string path;

        // typedefs
        struct Position {
            float x, y, z;
        };
        struct Rotation {
            float x, y, z;
        };
        typedef map<string, vector<tuple<int, Position, Rotation> > > Timesteps;
        typedef map<string, vector<float> > Speeds;

        // functions
        float getSpeed(string l0, string l1, string r0, string r1, int dt);
        string getFilename(string config, int time) {
            return path + "/" + config + "/" + to_string(time);
        };

    public:

        // constructors
        SpeedTest(string path);

        // class variables
        Timesteps timesteps;
        Speeds speeds;

        // functions
        vector<string> vectorize(string line);
        void calculateSpeeds();

};
