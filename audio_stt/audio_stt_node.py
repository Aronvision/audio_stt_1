#ros 설정
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# 기본 
import os
import threading
import time
import wave

# 추가
import numpy as np
from faster_whisper import WhisperModel
import speech_recognition as sr
import noisereduce as nr
import webrtcvad


class Audio_record:
    def __init__(self):
        '''
        요청 받았을 때 오디오를 스트리밍 하여 원하는 만큼 녹음하여 디노이즈
        '''
        # 기본 선언
        self.sample_rate = 16000
        self.chunk_duration_ms = 30 # 청크의 길이를 ms단위로 지정(VAD 로직에서 필요)
        self.vad_sec = 1 # n초 이상 말이 없을 경우 녹음 중지
        self.chunk_size = int(self.sample_rate * self.chunk_duration_ms / 1000) # 청크 크기를 샘플 단위로 계산
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone(sample_rate=self.sample_rate, chunk_size=self.chunk_size)
        self.buffer = []
        self.recording = False

        # VAD 초기화
        self.vad = webrtcvad.Vad(1) # 민감도 설정(0~3, 3이 제일 민감함)
        
        # 주변 소음 조정
        self.adjust_noise()

        print('Audio_record 초기화 성공')
        
    def adjust_noise(self):
        '''
        주변 소음 조정
        '''
        with self.microphone as source:
            print('주변 소음에 맞게 조정 중...')
            self.recognizer.adjust_for_ambient_noise(source)
            self.recognizer.energy_threshold += 100

    def record_start(self):
        '''녹음이 시작되는 함수'''
        if self.recording == False:
            self.record_thread = threading.Thread(target=self._record_start)
            self.record_thread.start()
    
    def _record_start(self):
        '''VAD 감지 조건으로 녹음이 게속되는 내부 함수'''
        self.recording = True
        self.buffer = []
        no_voice_target_cnt = (self.vad_sec*1000) # 녹음 목표 초를 ms로 변환
        no_voice_cnt = 0 # 위 변수와 비교할 cnt 설정
        with self.microphone as source:
            while self.recording:
                chunk = source.stream.read(self.chunk_size)
                self.buffer.append(chunk)
                if self._vad(chunk, self.sample_rate):
                    no_voice_cnt = 0
                else:
                    no_voice_cnt += self.chunk_duration_ms
                # vad가 일정 시간 감지 안되면 녹음 중지
                if no_voice_cnt >= no_voice_target_cnt:
                    self.recording = False

    def _vad(self, chunk, sample_rate):
        '''주어진 청크가 음성인지 여부를 반환하는 함수'''
        # 청크가 int16 형식인지 확인
        if isinstance(chunk, bytes):
            chunk = np.frombuffer(chunk, dtype=np.int16)
        # 청크가 10ms, 20ms, 30ms 길이인지 확인
        if len(chunk) != self.chunk_size:
            raise ValueError("Chunk size must be exactly 10ms, 20ms, or 30ms")
        return self.vad.is_speech(chunk.tobytes(), sample_rate)
        
    def record_stop(self, denoise_value):
        '''녹음이 종료되고 디노이징 과정을 거치는 함수'''
        # thread 종료하고, 끝날 때 까지 join으로 대기
        self.recording = False
        self.record_thread.join()
        # 버퍼를 하나의 오디오 데이터로 결합
        audio_data = np.frombuffer(b''.join(self.buffer), dtype=np.int16)
        sample_rate = self.microphone.SAMPLE_RATE
        return self._denoise_process(audio_data, sample_rate, denoise_value)

    
    def load_wav(self, path, denoise_value):
        '''wav파일을 불러와 디노이징 과정을 거치는 함수'''
        buffer = []
        with wave.open(path, 'rb') as wf:
            chunk_size = self.chunk_size
            data = wf.readframes(chunk_size)
            while data:
                buffer.append(data)
                data = wf.readframes(chunk_size)

        audio_data = np.frombuffer(b''.join(buffer), dtype=np.int16)
        sample_rate = wf.getframerate()
        return self._denoise_process(audio_data, sample_rate, denoise_value)

    
    def _denoise_process(self, audio_data, sample_rate, denoise_value):
        '''
        오디오를 받아 디노이징을 적용��고, 원본과 디노이즈값둘 둘 다 저장하고 반환한다.
        
        audio_data : int16 np 형식 오디오 데이터. chunk를 append하여 만들어진 buffer를 다음과 같이 처리한 예시) np.frombuffer(b''.join(self.buffer), dtype=np.int16)
        sample_rate : 샘플 레이트 입력
        denoise_value : 디노이즈 적용값 설정
        
        return: {'audio_denoise': audio_denoise, 'audio_noise': audio_noise, 'sample_rate': sample_rate}
        '''
        # 1. 노이즈 감소 처리
        denoise = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=denoise_value)
        buffer_denoise = [denoise.tobytes()] # 데이터를 다시 버퍼로 변환
        # 2. 노이즈 감소 없이
        noise = nr.reduce_noise(y=audio_data, sr=sample_rate, prop_decrease=0.0)
        buffer_noise = [noise.tobytes()] # 데이터를 다시 버퍼로 변환
        
        # 1. 노이즈 감소 파일 저장
        self._save_buffer_to_wav(buffer_denoise, self.microphone.SAMPLE_RATE, self.microphone.SAMPLE_WIDTH, 'input_denoise.wav')
        # 2. 노이즈 감소 없는 파일 저장
        self._save_buffer_to_wav(buffer_noise, self.microphone.SAMPLE_RATE, self.microphone.SAMPLE_WIDTH, 'input_noise.wav')
        
        # 오디오 소스 파일로 return
        audio_denoise = self._buffer_to_numpy(buffer_denoise, self.microphone.SAMPLE_RATE)
        audio_noise = self._buffer_to_numpy(buffer_noise, self.microphone.SAMPLE_RATE)

        return {'audio_denoise':audio_denoise, 'audio_noise':audio_noise, 'sample_rate':self.microphone.SAMPLE_RATE}


    def _buffer_to_numpy(self, buffer, sample_rate):
        '''buffer를 입력하면 whisper에서 추론 가능한 입력 형태의 오디오로 반환'''
        audio_data = np.frombuffer(b''.join(buffer), dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0  # Convert to float32        
        return audio_data
        

    def _save_buffer_to_wav(self, buffer, sample_rate, sample_width, filename):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # 모노
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(buffer))


class Custom_faster_whisper:
    def __init__(self):
        '''
        최대 4배 빠른 faster whisper를 ��용하여 cpu로 저장된 wav파일에 STT 수행
        '''
        # 환경 설정(Window 아나콘다 환경에서 아래 코드 실행 안하면 에러남)
        try: os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true"
        except Exception as e: print(f'os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "true" 실행해서 발생한 에러. 하지만 무시하고 진행: {e}')

        try: os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        except Exception as e: print(f'os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 실행해서 발생한 에러. 하지만 무시하고 진행: {e}')
        print('Cumtom_faster_whisper 초기화 성공')

    def set_model(self, model_name):
        '''
        모델 설정
        '''
        model_list = ['tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en', 'medium', 'medium.en', 'large-v1', 'large-v2', 'large-v3', 'large']
        if not model_name in model_list:
            model_name = 'base'
            print('모델 이름 잘못됨. base로 설정. 아래 모델 중 한가지 선택')
            print(model_list)
        self.model = WhisperModel(model_name, device="cpu", compute_type="int8")
        return model_name

    def run(self, audio, language=None):
        '''
        저장된 tmp.wav를 불러와서 STT 추론 수행

        audio : wav파일의 경로 or numpy로 변환된 오디오 파일 소스
        language : ko, en 등 언어 선택 가능. 선택하지 않으면 언어 분류 모델 내부적으로 수행함
        '''
        start = time.time()
        # 추론

        segments, info = self.model.transcribe(audio, beam_size=5, word_timestamps=True, language=language)
        # 결과 후처리
        dic_list = []
        for segment in segments:
            if segment.no_speech_prob > 0.6: continue # 말을 안했을 확률이 크다고 감지되면 무시
            for word in segment.words:
                _word = word.word
                _start = round(word.start, 2)
                _end = round(word.end, 2)
                dic_list.append([_word, _start, _end])
        # 시간 계산
        self.spent_time = round(time.time()-start, 2)
        
        # 텍스트 추출
        result_txt = self._make_txt(dic_list)
        print(result_txt)
        return dic_list, result_txt, self.spent_time

    def _make_txt(self, dic_list):
        '''
        [word, start, end]에서 word만 추출하여 txt로 반환
        '''
        result_txt = ''
        for dic in dic_list:
            txt = dic[0]
            result_txt = f'{result_txt}{txt}'
        return result_txt


class AudioSTTNode(Node):
    def __init__(self):
        super().__init__('audio_stt_node')
        self.publisher_ = self.create_publisher(String, 'stt_result', 10)
        self.get_logger().info('Audio STT Node has been started.')

        # 오디오 레코더 및 Whisper 모델 초기화
        self.audio_recorder = Audio_record()
        self.whisper = Custom_faster_whisper()
        model_name = self.whisper.set_model('base')  # 원하는 모델 선택

        # STT 처리 스레드 시작
        self.stt_thread = threading.Thread(target=self.stt_process)
        self.stt_thread.start()

    def stt_process(self):
        while rclpy.ok():
            # 녹음 시작
            self.get_logger().info("말씀하세요...")
            self.audio_recorder.record_start()
            
            try:
                # 녹음을 멈추기 위한 대기
                input("녹음을 멈추려면 Enter를 누르세요...")
            except KeyboardInterrupt:
                pass
            
            # 녹음 중지 및 디노이즈 처리
            audio_result = self.audio_recorder.record_stop(denoise_value=0.5)
            
            # STT 실행
            self.get_logger().info("음성을 텍스트로 변환 중...")
            _, result_text, spent_time = self.whisper.run(audio_result['audio_denoise'], language='ko')
            
            # 결과 퍼블리시
            msg = String()
            msg.data = result_text
            self.publisher_.publish(msg)
            
            self.get_logger().info(f"텍스트: {result_text}")
            self.get_logger().info(f"소요 시간: {spent_time}초")

            # 루프 계속 여부 확인
            try:
                cont = input("계속하시겠습니까? (y/n): ")
                if cont.lower() != 'y':
                    rclpy.shutdown()
                    break
            except KeyboardInterrupt:
                rclpy.shutdown()
                break

def main(args=None):
    rclpy.init(args=args)
    audio_stt_node = AudioSTTNode()
    rclpy.spin(audio_stt_node)
    audio_stt_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
