function d = EuclideanDistance(a,b)

if (nargin ~= 2)
    b = a;
end

if (size(a, 2) ~= size(b, 2))
   error('A and B should be of same dimensionality');
end

aa = sum(a .* a, 2); 
bb = sum(b .* b, 2); 
ab = a * b'; 
d = sqrt(abs(repmat(aa, [1 size(bb, 1)]) + repmat(bb', [size(aa, 1) 1]) - 2 * ab));