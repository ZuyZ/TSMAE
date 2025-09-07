#include <systemc.h>
#include <vector>
#include <fstream>
#include <iostream>
#include "new_TSMAE.h"

SC_MODULE(TestbenchTSMAE) {
    // Clock & Reset
    sc_clock        clk;
    sc_signal<bool> rst;

    // Data & handshake (theo cách bind của bạn)
    sc_signal<float> x_sig;
    sc_signal<bool>  rdy_to_receiv;    // TB → DUT
    sc_signal<bool>  is_rdy_to_receiv; // DUT → TB
    sc_signal<bool>  rdy_to_send;      // TB → DUT
    sc_signal<bool>  is_rdy_to_send;   // DUT → TB
    sc_signal<bool>  error_sig;
    sc_signal<float> x_recon_sig;
    sc_signal<bool> is_rdy_to_output;
    sc_signal<bool> rdy_to_output;
	std::vector<float> seq;
    // DUT instance
    TSMAE* tsmae;

    SC_CTOR(TestbenchTSMAE)
    : clk("clk", 10, SC_NS)
    {
        tsmae = new TSMAE("tsmae");
        tsmae->clk(clk);
        tsmae->rst(rst);
        tsmae->x(x_sig);
      	tsmae->is_rdy_to_output(rdy_to_output);
      	tsmae->rdy_to_output(is_rdy_to_output);
        // bind handshake swap style
        tsmae->rdy_to_receiv   ( is_rdy_to_send    );
        tsmae->is_rdy_to_receiv( rdy_to_send       );
        tsmae->rdy_to_send     ( is_rdy_to_receiv  );
        tsmae->is_rdy_to_send  ( rdy_to_receiv     );
        tsmae->x_recon         ( x_recon_sig       );
      	tsmae->load_weights();
		
        std::ifstream fin("normal_sample0.txt");
        float v;
        while (fin >> v) seq.push_back(v);
        fin.close();

        SC_THREAD(send_process);
      	
        sensitive << clk;
      SC_THREAD(receiv_process);
      	
        sensitive << clk;
       
    }

    void send_process() {
        // 1) đọc chuỗi input từ file
        
        // 2) RESET giữ 2 cạnh clock
        rst.write(true);
      is_rdy_to_send.write(0);
          is_rdy_to_output.write(0);
      
        wait(clk.posedge_event());
        wait(clk.posedge_event());
        rst.write(false);
        wait(clk.posedge_event());
		if (seq.empty()) {
            std::cerr << "No input data found!\n";
            sc_stop();
            return;
        }
        // 3) gửi từng giá trị trong seq
        for (int i = 0; i<seq.size(); i ++) {
            // chờ DUT ready to 
          x_sig.write(seq[i]);
          is_rdy_to_send.write(1);
          is_rdy_to_output.write(1);
          do { wait(clk.posedge_event()); }
            while (!rdy_to_send.read());
          is_rdy_to_send.write(0);              
        }      
    }
  
  void receiv_process()
  {
    for (int i = 0; i<seq.size(); i ++) {
		is_rdy_to_receiv.write(true);
            // chờ DUT xử lý xong
            do { wait(clk.posedge_event()); }
            while (!rdy_to_receiv.read());
			is_rdy_to_receiv.write(0);
            // in kết quả
            std::cout << "[" << sc_time_stamp() << "] "
                      << "x_recon["<<i<<"] = " << x_recon_sig.read()
                      << std::endl;

            // ACK lại DUT
      }
        sc_stop();
  }
};

int sc_main(int argc, char* argv[]) {
    TestbenchTSMAE tb("tb");
    sc_start();
    return 0;
}
