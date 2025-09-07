#include <systemc.h>
#include <iostream>

SC_MODULE(MemoryModule) {
    sc_in<bool> clk;
    sc_in<bool> rst;

    sc_in<bool> rdy_to_receiv;
    sc_in<bool> rdy_to_send;
    sc_in<float> last_ht[10]; // z input

    sc_out<float> latent[10]; // output vector
    sc_out<bool> is_rdy_to_receiv;
    sc_out<bool> is_rdy_to_send;

    float mem_item[20][10];       // memory items: m_i
    float last_ht_reg[10];        // copy of input z
    float sim[20];                // zᵗm_i
    float q[20];                  // softmax weights
    float latent_reg[10];         // output latent vector

    SC_CTOR(MemoryModule) {
        SC_CTHREAD(process, clk.pos());
        async_reset_signal_is(rst, true);
    }

    void process();
    float exp_approx(float x);
};

// ============================
// Định nghĩa bên ngoài module
// ============================

void MemoryModule::process() {
    while (true) {
        if (rst.read()) {
            
            for (int i = 0; i < 10; i++) {
                latent[i].write(0.0f);
                wait(); // mỗi ghi = 1 chu kỳ
            }
        } else {
            // --- chờ sẵn sàng nhận
            is_rdy_to_receiv.write(1);
            do {
                wait();
            } while (!rdy_to_receiv.read());
            is_rdy_to_receiv.write(0);

            // --- đọc z vào last_ht_reg
            for (int i = 0; i < 10; i++) {
                last_ht_reg[i] = last_ht[i].read();
                wait(); // mỗi read = 1 chu kỳ
            }

            // --- tính zᵗ m_i và exp(sim)
            float sum_exp = 0.0f;
            for (int i = 0; i < 20; i++) {
                sim[i] = 0.0f;
                wait(); // gán = 0.0 = 1 chu kỳ

                for (int j = 0; j < 10; j++) {
                    sim[i] += last_ht_reg[j] * mem_item[i][j];
                    wait(); // mỗi phép nhân + cộng = 1 chu kỳ
                }

                sim[i] = exp_approx(sim[i]);
                wait(); // tính exp = 1 chu kỳ

                sum_exp += sim[i];
                wait(); // cộng vào tổng = 1 chu kỳ
            }

            // --- softmax
            for (int i = 0; i < 20; i++) {
                q[i] = sim[i] / sum_exp;
                wait(); // mỗi phép chia = 1 chu kỳ
            }

            // --- recombine latent = ∑ q_i * m_i
            for (int j = 0; j < 10; j++) {
                latent_reg[j] = 0.0f;
                wait(); // khởi tạo = 1 chu kỳ

                for (int i = 0; i < 20; i++) {
                    latent_reg[j] += q[i] * mem_item[i][j];
                    wait(); // mỗi phép nhân + cộng = 1 chu kỳ
                }

                latent[j].write(latent_reg[j]);
                wait(); // mỗi write = 1 chu kỳ
            }

            // --- chờ gửi
            is_rdy_to_send.write(1);
            do {
                wait();
            } while (!rdy_to_send.read());
            is_rdy_to_send.write(0);
        }

        wait(); // kết thúc vòng lặp
    }
}

float MemoryModule::exp_approx(float x) {
    float result = 1.0f;
    float term = 1.0f;
    for (int i = 1; i <= 10; ++i) {
        term = term * x / i;
        result += term;
    }
    return result;
}
