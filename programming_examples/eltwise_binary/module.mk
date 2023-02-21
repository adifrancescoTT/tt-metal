ELTWISE_BINARY_EXAMPLE_SRC = programming_examples/eltwise_binary/eltwise_binary.cpp

ELTWISE_BINARY_EXAMPLES_DEPS = $(PROGRAMMING_EXAMPLES_OBJDIR)/eltwise_binary.d

-include $(ELTWISE_BINARY_EXAMPLES_DEPS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_TESTDIR)/eltwise_binary
$(PROGRAMMING_EXAMPLES_TESTDIR)/eltwise_binary: $(PROGRAMMING_EXAMPLES_OBJDIR)/eltwise_binary.o $(BACKEND_LIB) $(LL_BUDA_LIB) $(VERIF_LIB)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -o $@ $^ $(LDFLAGS) $(PROGRAMMING_EXAMPLES_LDFLAGS)

.PRECIOUS: $(PROGRAMMING_EXAMPLES_OBJDIR)/eltwise_binary.o
$(PROGRAMMING_EXAMPLES_OBJDIR)/eltwise_binary.o: $(ELTWISE_BINARY_EXAMPLE_SRC)
	@mkdir -p $(@D)
	$(CXX) $(CFLAGS) $(CXXFLAGS) $(PROGRAMMING_EXAMPLES_INCLUDES) -c -o $@ $<
