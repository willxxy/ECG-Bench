import argparse

from ecg_bench.utils.file_manager import FileManager


class MixECG:
    """Main class for mixing external datasets to base datas"""

    def __init__(self, args: argparse.Namespace, fm: FileManager):
        self.args = args
        self.fm = fm

    def mix_data(self):
        list_of_jsons = self.parse_mix_data()
        print("Mixing data from: ", list_of_jsons)
        data = []
        for json_file in list_of_jsons:
            data.extend(self.fm.open_json(f"./data/{json_file}.json"))
        print("Total instances: ", len(data))
        print("Segment length: ", list_of_jsons[0].split("_")[-1])
        self.fm.save_json(data, f"./data/{'_'.join(self.args.mix_data.split(','))}_mixed_{list_of_jsons[0].split('_')[-1]}.json")

    def parse_mix_data(self):
        return self.args.mix_data.split(",")
