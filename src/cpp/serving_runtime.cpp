// ALC native codec, bit-exact frame decoding, and paced socket transport.
#include "alc_tans_cpp_codec.cpp"
#include <pybind11/stl.h>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <condition_variable>
#include <mutex>
#include <memory>
#include <chrono>
#include <thread>
#include <sys/socket.h>
#include <cerrno>
#ifdef __APPLE__
#include <pthread.h>
#include <sys/qos.h>
#endif

namespace alc {
using Clock = std::chrono::steady_clock;
void io_priority() {
#ifdef __APPLE__
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE,0);
#endif
}
struct IoPriorityGuard {
#ifdef __APPLE__
    qos_class_t previous;
    IoPriorityGuard():previous(QOS_CLASS_DEFAULT){pthread_get_qos_class_np(pthread_self(),&previous,nullptr);io_priority();}
    ~IoPriorityGuard(){pthread_set_qos_class_self_np(previous==QOS_CLASS_UNSPECIFIED?QOS_CLASS_DEFAULT:previous,0);}
#else
    IoPriorityGuard()=default;
#endif
};
double now() { return std::chrono::duration<double>(Clock::now().time_since_epoch()).count(); }
struct Bits {
    const uint8_t* p; size_t cursor; uint64_t left,cache=0; unsigned available=0;
    Bits(const uint8_t* data,size_t bytes,uint64_t bits):p(data),cursor((bits+7)/8),left(bits) {
        if(bits>uint64_t(bytes)*8)throw std::invalid_argument("bit count exceeds payload");
        unsigned padding=unsigned(cursor*8-bits);
        if(padding){cache=p[--cursor]>>padding;available=8-padding;}
    }
    inline uint32_t read(unsigned n) {
        if(!n)return 0;
        if(n>32 || left<n)throw std::invalid_argument("truncated bitstream");
        if(available<n && cursor>=4) {
            cursor-=4;uint32_t word;std::memcpy(&word,p+cursor,4);
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            word=__builtin_bswap32(word);
#endif
            cache|=uint64_t(word)<<available;available+=32;
        }
        while(available<n) {
            if(!cursor)throw std::invalid_argument("truncated tail bytes");
            cache|=uint64_t(p[--cursor])<<available;available+=8;
        }
        uint32_t value=uint32_t(cache & ((uint64_t(1)<<n)-1));
        cache>>=n;available-=n;left-=n;return value;
    }
};

struct Table {
    DecodeEntry e[256]; bool uniform=false,single=false;
    Table(const uint8_t* entries, unsigned count, unsigned max_symbol) {
        if (!count || count>256) throw std::invalid_argument("invalid alphabet");
        uint16_t spread[256],freq[256],next[256]; bool seen[256]={};
        unsigned pos=0,total=0;
        for(unsigned i=0;i<count;++i) {
            auto s=entries[3*i]; freq[i]=read_u16_le(entries+3*i+1);
            if(s>max_symbol || seen[s] || !freq[i]) throw std::invalid_argument("invalid frequency");
            seen[s]=true; total+=freq[i]; next[i]=freq[i];
            if(total>256) throw std::invalid_argument("frequency overflow");
            for(unsigned j=0;j<freq[i];++j) {spread[pos]=i;pos=(pos+163)&255;}
        }
        if(total!=256) throw std::invalid_argument("frequency sum must be 256");
        uniform=count==256;single=count==1;
        for(unsigned state=0;state<256;++state) {
            auto i=spread[state];unsigned v=next[i]++; unsigned n=8-bsr32(v);
            e[state]={entries[3*i],uint8_t(n),(v<<n)-256};
        }
    }
    template<class F> void decode(const uint8_t* data,size_t bytes,uint64_t bits,uint64_t n,F consume) const {
        if(single && bits==16 && bytes==2 && !data[0] && !data[1]) {
            for(uint64_t i=n;i--;)consume(i,e[0].symbol);
            return;
        }
        // A 256-symbol, 8-bit table has frequency one for every symbol.
        // Its emitted states are byte aligned and independent: no ANS state
        // dependency is needed. Decode the EXISTING stream, byte-for-byte.
        if(uniform && bits==(n+2)*8 && bytes==n+2) {
            if(data[0] || data[n+1]) throw std::invalid_argument("invalid uniform ANS boundary states");
            for(uint64_t i=0;i<n;++i) consume(i,e[data[i+1]].symbol);
            return;
        }
        Bits reader(data,bytes,bits);auto original=reader.read(8),state=reader.read(8);
        for(uint64_t i=n;i--;) {auto entry=e[state];state=entry.new_state+reader.read(entry.nb_bits);consume(i,entry.symbol);}
        if(reader.left || state!=original) throw std::invalid_argument("invalid final ANS state");
    }
};
py::array_t<float> decode_frame(py::buffer input,py::buffer reference) {
    auto f=input.request(),r=reference.request();
    if(f.ndim!=1 || f.itemsize!=1 || f.strides[0]!=1 || f.size<17) throw std::invalid_argument("contiguous frame required");
    const auto* p=static_cast<const uint8_t*>(f.ptr); unsigned path=p[4];uint64_t n=read_u32_le(p+5);
    if(std::memcmp(p,"ALA1",4) || path>2 || read_u64_le(p+9)!=uint64_t(f.size-17)) throw std::invalid_argument("invalid frame header");
    if((r.itemsize!=4 || r.ndim!=1 || r.strides[0]!=4 || uint64_t(r.size)!=n)) throw std::invalid_argument("reference shape mismatch");
    py::array_t<float> out(n);auto* dst=reinterpret_cast<uint32_t*>(out.mutable_data());
    auto* ref=static_cast<const uint32_t*>(r.ptr);
    py::gil_scoped_release release;
    const uint8_t* body=p+17;size_t size=f.size-17;
    if(!path) {
        size_t offset=0;
        for(unsigned c=0;c<4;++c) {
            if(offset+8>size) throw std::invalid_argument("truncated column length");
            size_t bytes=read_u64_le(body+offset);offset+=8;
            if(bytes>size-offset || bytes<27) throw std::invalid_argument("truncated column");
            auto* q=body+offset;unsigned count=read_u16_le(q+13);uint64_t bits=read_u64_le(q+15);size_t stream=read_u32_le(q+23),head=27+3*count;
            if(std::memcmp(q,"BTAN",4) || q[12]!=8 || read_u64_le(q+4)!=n || head>bytes || stream!=bytes-head) throw std::invalid_argument("invalid byte column");
            if(n) {Table table(q+27,count,255);table.decode(q+head,stream,bits,n,[&](uint64_t i,uint8_t s){if(c==0)dst[i]=s;else dst[i]|=uint32_t(s)<<(8*c);});}
            else if(count || bits || stream) throw std::invalid_argument("invalid empty column");
            offset+=bytes;
        }
        if(offset!=size) throw std::invalid_argument("column trailer");
        for(uint64_t i=0;i<n;++i) dst[i]=rotr1_u32(dst[i]);
    } else {
        if(size<46 || std::memcmp(body,kMagic,4) || read_u64_le(body+4)!=n || body[12]!=8 || read_u64_le(body+14)!=n) throw std::invalid_argument("invalid delta frame");
        unsigned count=body[13];size_t head=46+3*count,jbytes=read_u32_le(body+38),mbytes=read_u32_le(body+42);
        if(count>65 || head>size || jbytes>size-head || mbytes!=size-head-jbytes) throw std::invalid_argument("delta sizes mismatch");
        if(n && count==1 && body[46]==32 && read_u16_le(body+47)==256 &&
           jbytes==2 && read_u64_le(body+22)==16 && !body[head] && !body[head+1] &&
           !mbytes && !read_u64_le(body+30)) {
            std::memcpy(dst,ref,n*4);return out; // exact identity in both mappings
        }
        Bits mantissa(body+head+jbytes,mbytes,read_u64_le(body+30));
        if(n) {
            Table table(body+46,count,64);
            table.decode(body+head,jbytes,read_u64_le(body+22),n,[&](uint64_t i,uint8_t symbol){
                uint32_t value=path==1?pcmap_u32(ref[i]):rotl1_u32(ref[i]);
                if(symbol>32) {unsigned b=symbol-33;value+=(uint32_t(1)<<b)+mantissa.read(b);}
                else if(symbol<32) {unsigned b=31-symbol;value-=(uint32_t(1)<<b)+mantissa.read(b);}
                dst[i]=path==1?pcmap_inverse_u32(value):rotr1_u32(value);
            });
        } else if(count || jbytes || mbytes) throw std::invalid_argument("invalid empty delta");
        if(mantissa.left) throw std::invalid_argument("mantissa trailer");
    }
    return out;
}
py::array_t<uint8_t> pack_wire(py::buffer input,const std::vector<std::array<uint64_t,3>>& fragments) {
    auto f=input.request();if(f.ndim!=1 || f.itemsize!=4 || f.strides[0]!=4) throw std::invalid_argument("FP32 buffer required");
    size_t bytes=f.size*4;
    for(auto a:fragments) {if(a[0]>uint64_t(f.size) || a[1]>uint64_t(f.size)-a[0] || a[2]>1) throw std::invalid_argument("BF16 span/rule invalid");bytes+=a[1]*2;}
    py::array_t<uint8_t> result(bytes);auto* out=result.mutable_data();auto* src=static_cast<const uint32_t*>(f.ptr);
    py::gil_scoped_release release;
    std::memcpy(out,src,f.size*4);size_t pos=f.size*4;
    for(auto a:fragments) for(uint64_t j=0;j<a[1];++j) {
        uint32_t u=src[a[0]+j];uint16_t b;
        if(!a[2]) b=(u&0x7fffffff)>0x7f800000?0x7fc0:uint16_t((u+0x7fff+((u>>16)&1))>>16);
        else b=uint16_t(u>>16);
        out[pos++]=b&255;out[pos++]=b>>8;
    }
    return result;
}
void send_all(int fd,const uint8_t* p,size_t n) {
    while(n) {
#ifdef MSG_NOSIGNAL
        auto k=::send(fd,p,n,MSG_NOSIGNAL);
#else
        auto k=::send(fd,p,n,0);
#endif
        if(k<0 && errno==EINTR)continue;
        if(k<=0)throw std::runtime_error("send failed errno="+std::to_string(errno));
        p+=k;n-=k;
    }
}
struct Sender {
    double rate,deadline=0; explicit Sender(double gbps):rate(gbps*1e9/8) {if(!(rate>0) || !std::isfinite(rate))throw std::invalid_argument("invalid rate");}
    std::array<double,4> send(int fd,py::buffer data,double ready_at=-1) {
        auto v=data.request();if(v.ndim!=1 || v.itemsize!=1 || v.strides[0]!=1)throw std::invalid_argument("contiguous byte payload required");
        auto* p=static_cast<const uint8_t*>(v.ptr);size_t n=v.size;uint8_t header[8];for(int j=0;j<8;++j)header[7-j]=uint64_t(n)>>(8*j);
        py::gil_scoped_release release;
        IoPriorityGuard io_guard;double begin=now(),sleep=0,syscall=0;
        if(ready_at<0)ready_at=begin;
        if(!std::isfinite(ready_at) || ready_at>begin+0.001)throw std::invalid_argument("invalid group ready timestamp");
        // A ready group models a queued NIC payload. Keep absolute service
        // deadlines across its chunks; CPU wakeup lateness is not a lower link
        // rate. A later group cannot borrow service before its decode finishes.
        deadline=std::max(deadline,ready_at);send_all(fd,header,8);
        const size_t block=256*1024;
        for(size_t off=0;off<n;off+=block) {
            size_t k=std::min(block,n-off);
            deadline+=double(k)/rate;
            if(double delay=deadline-now();delay>0) {auto t=now();std::this_thread::sleep_until(Clock::time_point(std::chrono::duration_cast<Clock::duration>(std::chrono::duration<double>(deadline))));sleep+=now()-t;}
            auto t=now();send_all(fd,p+off,k);syscall+=now()-t;
        }
        return {begin,now(),sleep,syscall};
    }
};
void recv_all(int fd,uint8_t* p,size_t n) {while(n){auto k=::recv(fd,p,n,0);if(k<0&&errno==EINTR)continue;if(k<=0)throw std::runtime_error("truncated recv errno="+std::to_string(errno));p+=k;n-=k;}}
std::array<double,2> recv_frame(int fd,py::buffer output) {
    auto v=output.request();if(v.readonly||v.ndim!=1||v.itemsize!=1||v.strides[0]!=1)throw std::invalid_argument("contiguous output required");
    auto* p=static_cast<uint8_t*>(v.ptr);size_t n=v.size;uint8_t header[8];
    py::gil_scoped_release release;
    recv_all(fd,header,8);uint64_t size=0;for(auto c:header)size=(size<<8)|c;if(size!=n)throw std::runtime_error("wire size mismatch");
    double first=0;
    for(size_t off=0;off<n;) {auto k=::recv(fd,p+off,std::min(size_t(256*1024),n-off),0);if(k<0&&errno==EINTR)continue;if(k<=0)throw std::runtime_error("truncated payload");if(!first)first=now();off+=k;}
    return {first,now()};
}
// Entire receive+verify pipeline stays native while the Python GIL is released.
struct Receiver {
    std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> manifest{EVP_MD_CTX_new(),EVP_MD_CTX_free};
    std::vector<uint64_t> sizes;
    std::vector<std::array<uint8_t,32>> expected;
    std::vector<uint8_t> buffers[2];
    Receiver() {if(!manifest || EVP_DigestInit_ex(manifest.get(),EVP_sha256(),nullptr)!=1)throw std::runtime_error("SHA256 initialization failed");}
    void prepare(std::vector<uint64_t> spans,std::vector<std::string> hashes) {
        if(spans.empty() || spans.size()!=hashes.size())throw std::invalid_argument("invalid receive manifest");
        sizes=std::move(spans);expected.clear();size_t largest=0;
        for(size_t i=0;i<sizes.size();++i) {
            if(!sizes[i] || hashes[i].size()!=64)throw std::invalid_argument("invalid receive span/hash");
            std::array<uint8_t,32> digest;
            auto nibble=[](char c)->uint8_t{if(c>='0'&&c<='9')return c-'0';if(c>='a'&&c<='f')return c-'a'+10;throw std::invalid_argument("invalid SHA hex");};
            for(size_t j=0;j<32;++j)digest[j]=(nibble(hashes[i][2*j])<<4)|nibble(hashes[i][2*j+1]);
            expected.push_back(digest);largest=std::max(largest,size_t(sizes[i]));
        }
        // Allocate and write every page before the decode/transfer timer.
        py::gil_scoped_release release;
        for(auto& b:buffers){b.resize(largest);std::fill(b.begin(),b.end(),uint8_t(0));}
    }
    void clear_buffers() {py::gil_scoped_release release;for(auto& b:buffers)std::vector<uint8_t>().swap(b);}
    py::dict receive(int fd) {
        double first=0,first_checked=0,last_arrival=0,finished=0;
        uint64_t total=0;size_t verified=0;
        std::array<uint8_t,32> final_digest{};
        {
            py::gil_scoped_release release;
            io_priority();
            struct Work {size_t slot,index;};
            std::mutex mutex;std::condition_variable available,work_ready;
            std::queue<size_t> free;free.push(0);free.push(1);
            std::queue<Work> work;bool done=false;std::exception_ptr error;
            auto verify_loop=[&]{
                io_priority();
                try {
                    for(;;) {
                        Work item;
                        {std::unique_lock<std::mutex> lock(mutex);work_ready.wait(lock,[&]{return done||!work.empty();});if(work.empty())break;item=work.front();work.pop();}
                        uint8_t digest[32];
                        if(!SHA256(buffers[item.slot].data(),sizes[item.index],digest) || !std::equal(digest,digest+32,expected[item.index].begin()))throw std::runtime_error("client/source SHA mismatch");
                        {std::lock_guard<std::mutex> lock(mutex);
                            finished=now();if(item.index==0)first_checked=finished;++verified;free.push(item.slot);
                        }available.notify_one();
                    }
                } catch(...) {{std::lock_guard<std::mutex> lock(mutex);error=std::current_exception();}available.notify_all();}
            };
            std::thread verifier0(verify_loop),verifier1(verify_loop);
            try {
                for(size_t i=0;i<sizes.size();++i) {
                    size_t slot;
                    {std::unique_lock<std::mutex> lock(mutex);available.wait(lock,[&]{return error||!free.empty();});if(error)std::rethrow_exception(error);slot=free.front();free.pop();}
                    uint8_t header[8];recv_all(fd,header,8);uint64_t bytes=0;for(auto c:header)bytes=(bytes<<8)|c;
                    if(bytes!=sizes[i])throw std::runtime_error("wire chunk geometry mismatch");
                    for(size_t off=0;off<bytes;) {
                        auto k=::recv(fd,buffers[slot].data()+off,std::min(size_t(1024*1024),size_t(bytes)-off),0);
                        if(k<0&&errno==EINTR)continue;if(k<=0)throw std::runtime_error("truncated native receive");
                        if(!first)first=now();off+=k;
                    }
                    total+=bytes;last_arrival=now();
                    {std::lock_guard<std::mutex> lock(mutex);work.push({slot,i});}work_ready.notify_one();
                }
            } catch(...) {{std::lock_guard<std::mutex> lock(mutex);if(!error)error=std::current_exception();}}
            {std::lock_guard<std::mutex> lock(mutex);done=true;}work_ready.notify_all();verifier0.join();verifier1.join();
            if(error)std::rethrow_exception(error);
            if(verified!=sizes.size())throw std::runtime_error("incomplete native verification");
            // Payload hashes have all been independently verified. Fold the
            // expected (therefore equal actual) hashes in wire order; verifier
            // completion order must never change the query digest.
            for(size_t i=0;i<sizes.size();++i) {
                uint8_t length[8];for(unsigned j=0;j<8;++j)length[7-j]=sizes[i]>>(8*j);
                if(EVP_DigestUpdate(manifest.get(),length,8)!=1 || EVP_DigestUpdate(manifest.get(),expected[i].data(),32)!=1)throw std::runtime_error("manifest SHA update failed");
            }
            std::unique_ptr<EVP_MD_CTX,decltype(&EVP_MD_CTX_free)> copy(EVP_MD_CTX_new(),EVP_MD_CTX_free);unsigned count=0;
            if(!copy || EVP_MD_CTX_copy_ex(copy.get(),manifest.get())!=1 || EVP_DigestFinal_ex(copy.get(),final_digest.data(),&count)!=1 || count!=32)throw std::runtime_error("manifest SHA final failed");
            finished=now();
        }
        const char* hex="0123456789abcdef";std::string digest;
        for(auto b:final_digest){digest+=hex[b>>4];digest+=hex[b&15];}
        py::dict result;result["first"]=first;result["first_checked"]=first_checked;result["last_arrival"]=last_arrival;result["finished"]=finished;result["bytes"]=total;result["verified"]=verified;result["digest"]=digest;return result;
    }
};

}
PYBIND11_MODULE(alc_tans_cpp_codec,m) {
    m.def("tans_encode",[](py::bytes data,uint32_t log){if(py::len(data)==0)throw std::invalid_argument("empty codec chunks are not stored");return tans_encode_bytes_cpp(data,log);},py::arg("data"),py::arg("table_log")=8);
    m.def("tans_decode",&tans_decode_bytes_cpp);
    m.def("encode_pcdelta_tans_f32_buffer",&encode_pcdelta_tans_f32_buffer_cpp,py::arg("reference"),py::arg("target"),py::arg("table_log")=8);
    m.def("encode_rotdelta_tans_f32_buffer",&encode_rotdelta_tans_f32_buffer_cpp,py::arg("reference"),py::arg("target"),py::arg("table_log")=8);
    m.def("decode_frame",&alc::decode_frame);
    m.def("pack_wire",&alc::pack_wire);
    m.def("clock",&alc::now);
    m.attr("openssl_version")=OpenSSL_version(OPENSSL_VERSION);
    m.def("recv_frame",&alc::recv_frame);
    py::class_<alc::Receiver>(m,"Receiver").def(py::init<>()).def("prepare",&alc::Receiver::prepare).def("receive",&alc::Receiver::receive).def("clear_buffers",&alc::Receiver::clear_buffers);
    py::class_<alc::Sender>(m,"Sender").def(py::init<double>()).def("send",&alc::Sender::send,py::arg("fd"),py::arg("data"),py::arg("ready_at")=-1.0).def_readonly("service_clock_deadline",&alc::Sender::deadline);
}
