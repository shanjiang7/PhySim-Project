#include <iostream>
#include <vector>

// Create structures for points and edges
struct point{
    double x, y;
};

struct connector{
    int p1, p2; // indices of the points that the connector is attatched to
};

int main() {
    // create a square mesh where side_length is the length side of the entire square
    // n_seg is the number of segments the side is split up into
    // each side will have n_seg+1 points
    double side_length = 2;
    int n_seg = 2;
    double seg_length = side_length / n_seg;

    std::vector<point>points;
    std::vector<connector>connectors;

    // Generate grid points on the square
    for (int i = 0; i < n_seg + 1; ++i){
        for (int j = 0; j < n_seg + 1; ++j){
            points.push_back({i * seg_length, j * seg_length});
        }
    }

    // Add connectors for all the points (horizontal, vertical, and diagonals)
    for (int i = 0; i < n_seg + 1; ++i){
        for (int j = 0; j < n_seg + 1; ++j){
            int idx = i * (n_seg + 1) + j;
            //Connect right
            if (j < n_seg)
                connectors.push_back({idx, idx+1});
            //Connect top
            if (i < n_seg)
                connectors.push_back({idx, idx + n_seg + 1});
            //Connect top-right
            if (j < n_seg && j < n_seg){
                if (idx + n_seg + 1 + 1 <= (n_seg+1)*(n_seg+1))
                    connectors.push_back({idx, idx + n_seg + 1 + 1});
            }
                
            //Connect top-left
            if (j > 0 && i < n_seg)
                connectors.push_back({idx, idx + n_seg + 1 - 1});
        }
    }

    //Output the points
    std::cout << "Points:\n"; 
    for(int i = 0; i<points.size(); ++i){
        std::cout << i << ": (" << points[i].x << ", " << points[i].y << ")\n"; 
    }

    //Ouput the edges
    std::cout << "\n Connectors: \n";
    for (const auto& connector: connectors){
        std::cout << connector.p1 << "<->" << connector.p2 << "\n";
    }

    return 0;


}