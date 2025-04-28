
//THIS IS FOR HAVING POINTS AND CONNECTORS AS THEIR OWN STRUCTURES 
// Note from Nina: below this is a commented out portion that is exactly the same code and does the same thing except
// that it will make points and connectors 1D vectors instead of structured objects. 
// I included both because I originally did it the first way with having them as structured objects
// but then realized that to implement with MUDA later it might be easier to have them as 1D vectors. 

// #include <iostream>
// #include <vector>

// // Create structures for points and edges
// struct point{
//     double x, y;
// };

// struct connector{
//     int p1, p2; // indices of the points that the connector is attatched to
// };

// int main() {
//     // create a square mesh where side_length is the length side of the entire square
//     // n_seg is the number of segments the side is split up into
//     // each side will have n_seg+1 points
//     double side_length = 2;
//     int n_seg = 2;
//     double seg_length = side_length / n_seg;

//     std::vector<point>points;
//     std::vector<connector>connectors;

//     // Generate grid points on the square
//     for (int i = 0; i < n_seg + 1; ++i){
//         for (int j = 0; j < n_seg + 1; ++j){
//             points.push_back({i * seg_length, j * seg_length});
//         }
//     }

//     // Add connectors for all the points (horizontal, vertical, and diagonals)
//     for (int i = 0; i < n_seg + 1; ++i){
//         for (int j = 0; j < n_seg + 1; ++j){
//             int idx = i * (n_seg + 1) + j;
//             //Connect right
//             if (j < n_seg)
//                 connectors.push_back({idx, idx+1});
//             //Connect top
//             if (i < n_seg)
//                 connectors.push_back({idx, idx + n_seg + 1});
//             //Connect top-right
//             if (j < n_seg && j < n_seg){
//                 if (idx + n_seg + 1 + 1 <= (n_seg+1)*(n_seg+1))
//                     connectors.push_back({idx, idx + n_seg + 1 + 1});
//             }
                
//             //Connect top-left
//             if (j > 0 && i < n_seg)
//                 connectors.push_back({idx, idx + n_seg + 1 - 1});
//         }
//     }

//     //Output the points
//     std::cout << "Points:\n"; 
//     for(int i = 0; i<points.size(); ++i){
//         std::cout << i << ": (" << points[i].x << ", " << points[i].y << ")\n"; 
//     }

//     //Ouput the edges
//     std::cout << "\n Connectors: \n";
//     for (const auto& connector: connectors){
//         std::cout << connector.p1 << "<->" << connector.p2 << "\n";
//     }

//     return 0;

// }


//THIS IS FOR HAVING POINTS AND CONNECTORS AS 1D VECTORS
// Box will be centered at (0,0) for this script


#include <iostream>
#include <vector>

int main() {
    double side_length = 2;
    int n_seg = 2;
    double seg_length = side_length / n_seg;

    std::vector<double> points;   // 1D vector for x, y pairs
    std::vector<int> connectors;  // 1D vector for edge connections

    int dim = n_seg + 1;
    points.reserve(dim * dim * 2); // preallocate space for points
    connectors.reserve(2 * n_seg * (n_seg + 1) + 4 * n_seg * n_seg); // estimate edges

    // Generate grid points
    for (int i = 0; i < dim; ++i){
        for (int j = 0; j < dim; ++j){
            points.push_back(-side_length/2 + i * seg_length); // x coordinate
            points.push_back(-side_length/2 + j * seg_length); // y coordinate
        }
    }

    // Generate edges (horizontal, vertical, and diagonals)
    for (int i = 0; i < dim; ++i){
        for (int j = 0; j < dim; ++j){
            int idx = i * dim + j;
            // Connect right
            if (j < n_seg) {
                connectors.push_back(idx);
                connectors.push_back(idx + 1);
            }
            // Connect up
            if (i < n_seg) {
                connectors.push_back(idx);
                connectors.push_back(idx + dim);
            }
            // Connect top-right diagonal
            if (i < n_seg && j < n_seg) {
                connectors.push_back(idx);
                connectors.push_back(idx + dim + 1);
            }
            // Connect top-left diagonal
            if (i < n_seg && j > 0) {
                connectors.push_back(idx);
                connectors.push_back(idx + dim - 1);
            }
        }
    }

    // Output the points
    std::cout << "Points:\n"; 
    for (int i = 0; i < points.size(); i += 2){
        std::cout << i/2 << ": (" << points[i] << ", " << points[i+1] << ")\n"; 
    }

    // Output the connectors
    std::cout << "\nConnectors:\n";
    for (int i = 0; i < connectors.size(); i += 2){
        std::cout << connectors[i] << " <-> " << connectors[i+1] << "\n";
    }

    return 0;
}
