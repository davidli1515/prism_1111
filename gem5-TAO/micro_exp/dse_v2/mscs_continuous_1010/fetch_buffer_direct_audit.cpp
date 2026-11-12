// Read-only numerical audit using the original training feature implementation.
#include "/scratch/gem5-TAO/concord_analytical/analytical_models.cpp"

int main(int argc,char** argv) {
    if(argc!=3)return 2;
    ConcordeTrace ct=parse_trace(argv[1]);
    if(ct.n_windows<=0)return 3;
    ArchConfig cfg{};cfg.l1_size_kb=64;cfg.l2_size_kb=1024;
    cfg.l1i_assoc=4;cfg.l2_assoc=16;
    std::vector<int32_t> lat;icache_sim(ct,cfg,lat);
    FILE* out=fopen(argv[2],"wx");if(!out)return 4;
    fprintf(out,"{\"windows\":%d,\"instructions\":%d,\"runs\":[",ct.n_windows,ct.size());
    bool first=true;
    for(int nb: {8,16,32,64})for(int dw:{8,9,12}) {
        std::vector<double> thr(ct.n_windows);double features[DIST_DIM];
        fetch_bufs_model(ct,nb,dw,lat,thr.data());
        encode_distribution(thr.data(),ct.n_windows,features);
        if(!first)fprintf(out,",");first=false;
        fprintf(out,"{\"nb\":%d,\"dw\":%d,\"throughput\":[",nb,dw);
        for(int i=0;i<ct.n_windows;i++)fprintf(out,"%s%.17g",i?",":"",thr[i]);
        fprintf(out,"],\"features\":[");
        for(int i=0;i<DIST_DIM;i++)fprintf(out,"%s%.17g",i?",":"",features[i]);
        fprintf(out,"]}");
    }
    fprintf(out,"]}\n");fclose(out);return 0;
}
