cl__1 = .5;
cl__2 = 1;

// Circle(id) = {startPoint, centerPoint, endPoint};

//
// Points for center and key points on circles
Point(1) = {0, 0, 0, cl__2};  // center
Point(2) = {10, 0, 0, cl__2};  // outer circle start
Point(3) = {-10, 0, 0, cl__2}; // outer circle end (opposite side)
Point(4) = {5, 0, 0, cl__1}; // inner circle start
Point(5) = {-5, 0, 0, cl__1}; // inner circle end

// Define outer circle
Circle(1) = {2, 1, 3};
Circle(2) = {3, 1, 2};

// Inner circle
Circle(3) = {4, 1, 5};
Circle(4) = {5, 1, 4};

// Line loops
Line Loop(1) = {1, 2};
Line Loop(2) = {3, 4};

// Define surface bounded by the loop
Plane Surface(1) = {1, 2};

// Assign physical groups for boundary and surface
Physical Curve("inner") = {1, 2};
Physical Curve("outer") = {3, 4};
Physical Surface("surface") = {1};
