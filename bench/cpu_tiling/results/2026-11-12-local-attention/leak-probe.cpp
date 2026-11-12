#include <cstdlib>
#include <vector>
#include <cmath>
#include <cassert>
extern "C" float* mim_attention(float*, float*, float*);
int main() {
 std::vector<float> q(8*128*64,0.01f), k(q), v(q);
 for(int i=0;i<20;++i) {
  auto out=mim_attention(q.data(),k.data(),v.data());
  for(int j=0;j<8*128*64;++j) assert(std::isfinite(out[j]) && std::abs(out[j]-.01f)<.0001f);
  std::free(out);
 }
}
