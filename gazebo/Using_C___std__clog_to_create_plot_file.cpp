// in initialization
std::clog << "SuccessfulGrabs TotalRuns Accuracy LearningRate maxLearningRate LSTMSize Last100Accuracy" << std::endl << std::flush;

// in OnUpdate(), at end of episode
			std::clog <<  
				successfulGrabs << " " << totalRuns << " " << 
				(float(successfulGrabs)/float(totalRuns)) << " " <<
				LearningRate << " " << maxLearningRate << " " << 
				LSTMSize << " " << (float(historyWins)/float(RUN_HISTORY)) <<
				std::endl << std::flush;
