#include <systemc.h>
#include <iostream>

SC_MODULE(Mem_de_driver) {
    sc_in<float> latent[10];
    sc_in<bool> clk;
    sc_in<bool> rst;
    sc_in<bool> rdy_to_receiv;
    sc_in<bool> rdy_to_send;
    sc_out<bool> is_rdy_to_send;
    sc_out<bool> is_rdy_to_receiv;
    sc_out<float> latent_series[10];

    float buffer[10];
    int count;
    int index;

    SC_CTOR(Mem_de_driver) {
        SC_CTHREAD(driver_process, clk.pos());
        async_reset_signal_is(rst, true);
    }

    void driver_process();
};

// ========== Implementation phần tách ra ngoài ==========

void Mem_de_driver::driver_process() {
    while (true) {
        if (rst.read()) {
            count = 0;
            index = 0;
            

            for (int i = 0; i < 10; i++) {
                buffer[i] = 0;
                latent_series[i].write(0);
                wait();  // mỗi bước reset = 1 chu kỳ
            }
        } else {
            count = 0;
            is_rdy_to_receiv.write(1);
            do {
                wait();  // chờ module trước sẵn sàng
            } while (!rdy_to_receiv.read());
            is_rdy_to_receiv.write(0);

            // Đọc latent vào buffer từng phần tử
            for (int i = 0; i < 10; i++) {
                buffer[i] = latent[i].read();
                wait();  // mỗi read = 1 chu kỳ
            }

            while (count < 140) {
                for (index = 0; index < 10; index++) {
                    latent_series[index].write(buffer[index]);
                    wait();  // mỗi write = 1 chu kỳ
                }

                is_rdy_to_send.write(1);
                do {
                    wait();  // chờ bên nhận đọc xong
                } while (!rdy_to_send.read());
                is_rdy_to_send.write(0);
                wait();  // giữ tín hiệu ổn định 1 chu kỳ

                count = count + 1;
                wait();  // mỗi lần phát xong +1 = 1 chu kỳ
            }
        }

        wait();  // cuối vòng lặp
    }
}
